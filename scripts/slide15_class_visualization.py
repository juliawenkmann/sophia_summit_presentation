import os, time, math, random, logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image

# -----------------------------
# 0) Runtime / device settings
# -----------------------------
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)

torch.backends.cudnn.benchmark = True  # let CuDNN pick fast kernels

# -----------------------------
# 1) Model + ImageNet metadata
# -----------------------------
def build_model(use_compile: bool = True):
    """
    Build an ImageNet model (GoogLeNet), disable aux classifiers (faster at eval),
    freeze params, switch to eval, optionally torch.compile (PyTorch >= 2.0).
    """
    weights = models.GoogLeNet_Weights.IMAGENET1K_V1
    net = models.googlenet(weights=weights, aux_logits=False).to(device).eval()
    for p in net.parameters(): p.requires_grad_(False)
    if use_compile and hasattr(torch, "compile"):
        try:
            net = torch.compile(net, mode="reduce-overhead", fullgraph=False)
        except Exception as e:
            print(f"[warn] torch.compile failed ({e}); continuing without compile.")

    categories = weights.meta["categories"]
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]

    return net, categories, mean, std

NET, CATEGORIES, IMAGENET_MEAN, IMAGENET_STD = build_model(use_compile=True)

def class_idx_from_name(name: str, categories=CATEGORIES) -> int:
    """
    Resolve a (partial) human label to ImageNet index.
    Tries exact match first, then substring contains.
    """
    name_l = name.lower()
    try:
        return categories.index(name_l)
    except ValueError:
        hits = [i for i, s in enumerate(categories) if name_l in s.lower()]
        if not hits:
            raise ValueError(f"'{name}' not found in ImageNet categories.")
        return hits[0]

# -----------------------------
# 2) Fourier parameterization
# -----------------------------
def radial_freq(h: int, w: int, dev: str):
    """Radial frequency grid for rFFT2 shapes."""
    fy = torch.fft.fftfreq(h, d=1.0).to(dev).reshape(h, 1)
    fx = torch.fft.rfftfreq(w, d=1.0).to(dev).reshape(1, w // 2 + 1)
    return torch.sqrt(fx * fx + fy * fy).clamp(min=1e-6)

def make_fft_params(h: int, w: int, decay_power: float):
    """
    Create real+imag tensor for rFFT2 spectrum and a *precomputed* inverse decay
    (1 / f**decay_power) for speed.
    """
    spec = torch.randn(1, 3, h, w // 2 + 1, 2, device=device, requires_grad=True)
    freqs = radial_freq(h, w, device)
    inv_decay = (1.0 / (freqs ** decay_power))  # [H, W//2+1]
    return spec, inv_decay

def spectrum_to_image(spec: torch.Tensor, inv_decay: torch.Tensor) -> torch.Tensor:
    """
    Convert learnable rFFT2 spectrum -> spatial RGB image in [0,1] with gradients.
    - spec: [1,3,H,W//2+1,2], real+imag in last dim
    - inv_decay: [H, W//2+1] (broadcasted)
    """
    complex_spec = torch.view_as_complex(spec)                    # [1,3,H,Wr]
    scaled = complex_spec * inv_decay                             # broadcast
    H, W = spec.shape[2], (spec.shape[3] - 1) * 2
    img = torch.fft.irfft2(scaled, s=(H, W), norm="ortho")        # [1,3,H,W]

    # simple contrast stabilization + squashing into [0,1]
    img = img / (img.std(dim=(-2, -1), keepdim=True) + 1e-8)
    img = torch.tanh(img) * 0.5 + 0.5
    return img

# -----------------------------
# 3) Regularizers / filters
# -----------------------------
def tv_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Total variation (anisotropic). x: [B, C, H, W]
    """
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return (dx.pow(2).mean() + dy.pow(2).mean())

def blur3(x: torch.Tensor) -> torch.Tensor:
    """3x3 box blur to suppress ringing."""
    return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

# -----------------------------
# 4) Robustness augmentations
# -----------------------------
def random_view(
    x: torch.Tensor,
    out_hw: Tuple[int, int] = (224, 224),
    jitter: int = 24,
    rot: float = 15,
    scale_range: Tuple[float, float] = (0.92, 1.08),
    flip: bool = True,
    noise_std: float = 0.02,
) -> torch.Tensor:
    """
    Stochastic view for transform-robust optimization (fully torch-based).
    x: [1,3,H,W] in [0,1]
    """
    _, _, H, W = x.shape

    # integer jitter via roll (sample on device)
    ox = int(torch.randint(-jitter, jitter + 1, (1,), device=device).item())
    oy = int(torch.randint(-jitter, jitter + 1, (1,), device=device).item())
    v = torch.roll(x, shifts=(ox, oy), dims=(3, 2))  # (W, H)

    # small affine (angle deg, translate px, scale)
    angle = float((torch.rand(1, device=device) * 2 * rot - rot).item())
    scale = float((torch.rand(1, device=device) * (scale_range[1] - scale_range[0]) + scale_range[0]).item())
    tx = int((torch.rand(1, device=device) * 0.12 - 0.06).item() * W)
    ty = int((torch.rand(1, device=device) * 0.12 - 0.06).item() * H)
    v = TF.affine(v, angle=angle, translate=[tx, ty], scale=scale,
                  shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR)

    if flip and torch.rand(()) < 0.5:
        v = TF.hflip(v)

    # resize to model input
    v = F.interpolate(v, size=out_hw, mode="bilinear", align_corners=False)

    # light gaussian noise, then normalize
    if noise_std > 0:
        v = (v + noise_std * torch.randn_like(v)).clamp(0, 1)
    v = (v - IMAGENET_MEAN) / IMAGENET_STD
    return v

# -----------------------------
# 5) Utilities
# -----------------------------
@torch.no_grad()
def to_pil(x: torch.Tensor) -> Image.Image:
    x = x.squeeze(0).clamp(0, 1).detach().cpu()
    return TF.to_pil_image(x)

def setup_logger(out_dir: Path, log_name="classviz") -> logging.Logger:
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(out_dir / f"{log_name}.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

def topk_from_logits(logits: torch.Tensor, k: int = 5):
    probs = logits.float().softmax(dim=-1)
    vals, idxs = probs.topk(k, dim=-1)
    labels = [CATEGORIES[i] for i in idxs.tolist()]
    return list(zip(labels, vals.tolist()))

# -----------------------------
# 6) Main optimization
# -----------------------------
@dataclass
class SynthConfig:
    class_name: str = "dumbbell"
    steps: int = 700
    n_views: int = 8
    size: int = 384
    lr: float = 0.08
    tv_w: float = 1e-4
    l2_w: float = 1e-6
    decay_power: float = 1.5
    blur_every: int = 60
    log_every: int = 25
    save_every: int = 100
    out_dir: str = "runs"
    seed: Optional[int] = 123
    use_amp: bool = True          # mixed precision on CUDA (faster)
    use_compile: bool = True      # torch.compile (if available)

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def synthesize_class(cfg: SynthConfig) -> Image.Image:
    """
    Maximize model score of ImageNet class using Fourier-parameterized image.
    Returns a final PIL image and saves periodic checkpoints / logs.
    """
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Refresh model if compile preference changes
    global NET
    NET, _, _, _ = build_model(use_compile=cfg.use_compile)

    target_idx = class_idx_from_name(cfg.class_name)
    out_root = Path(cfg.out_dir) / cfg.class_name.replace(" ", "_")
    out_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_root)

    logger.info(f"Class: '{cfg.class_name}' (index {target_idx}) | "
                f"steps={cfg.steps}, n_views={cfg.n_views}, size={cfg.size}, "
                f"lr={cfg.lr}, amp={cfg.use_amp}, compile={cfg.use_compile}")

    spec, inv_decay = make_fft_params(cfg.size, cfg.size, cfg.decay_power)
    opt = torch.optim.Adam([spec], lr=cfg.lr)

    scaler = torch.amp.GradScaler(enabled=(cfg.use_amp and device == "cuda"))
    device_type = "cuda" if device == "cuda" else "cpu"

    best_score = -float("inf")
    best_img: Optional[Image.Image] = None
    t0 = time.perf_counter()

    for step in range(1, cfg.steps + 1):
        # Build image from spectrum (keep this in fp32 to avoid complex/AMP quirks)
        img = spectrum_to_image(spec, inv_decay)

        # Occasional blur to tame ringing
        if cfg.blur_every and step % cfg.blur_every == 0:
            img = blur3(img)

        # Build a minibatch of random views
        views = [random_view(img) for _ in range(cfg.n_views)]
        batch = torch.cat(views, dim=0)  # [V,3,224,224]

        # Forward pass (AMP just for the model for speed)
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(cfg.use_amp and device == "cuda")):
            logits = NET(batch)                    # [V, 1000]
            cls_mean = logits[:, target_idx].mean()  # scalar

        # Priors (fp32)
        tv = tv_loss(img)
        l2 = ((img - 0.5) ** 2).mean()

        # We maximize class score => descend on negative of objective
        loss = cls_mean - cfg.tv_w * tv - cfg.l2_w * l2

        opt.zero_grad(set_to_none=True)
        scaler.scale(-loss).backward()
        scaler.step(opt)
        scaler.update()

        # Gentle spectrum damping prevents exploding amplitudes
        with torch.no_grad():
            spec.mul_(0.995)

        # ---- Logging / checkpoints ----
        if step % cfg.log_every == 0 or step == 1:
            # Use the batch we just computed to report top-1 (avg over views)
            avg_logits = logits.mean(dim=0)  # [1000]
            top1_label, top1_prob = topk_from_logits(avg_logits, k=1)[0]
            elapsed = time.perf_counter() - t0
            logger.info(
                f"step {step:4d}/{cfg.steps}  "
                f"score={cls_mean.item():.4f}  "
                f"tv={tv.item():.5f}  l2={l2.item():.5f}  "
                f"top1='{top1_label}' p={top1_prob:.3f}  "
                f"t/step~{elapsed/step:.3f}s"
            )

        # Track best (by robust mean class score)
        cur_score = float(cls_mean.item())
        if cur_score > best_score:
            best_score = cur_score
            best_img = to_pil(img)

        # Save periodic checkpoints
        if (cfg.save_every and step % cfg.save_every == 0) or step == cfg.steps:
            pil_img = to_pil(img)
            ckpt_path = out_root / f"{cfg.class_name.replace(' ','_')}_step{step:04d}.png"
            pil_img.save(ckpt_path)
            logger.info(f"saved checkpoint: {ckpt_path}")

    # Save final + best
    final_img = to_pil(spectrum_to_image(spec, inv_decay))
    final_path = out_root / f"{cfg.class_name.replace(' ','_')}_final.png"
    final_img.save(final_path)
    logger.info(f"saved final: {final_path}")

    if best_img is not None:
        best_path = out_root / f"{cfg.class_name.replace(' ','_')}_best.png"
        best_img.save(best_path)
        logger.info(f"saved best (by running score): {best_path}")

    return final_img

# -----------------------------
# 7) Example usage
# -----------------------------
if __name__ == "__main__":
    cfg = SynthConfig(
        class_name="dumbbell",
        steps=1000,
        n_views=8,
        size=448,
        lr=0.01,
        decay_power=0.5,
        tv_w=1e-4,
        l2_w=1e-6,
        blur_every=60,
        log_every=25,
        save_every=200,
        out_dir="runs",
        seed=123,
        use_amp=True,
        use_compile=True,
    )
    out = synthesize_class(cfg)
    out.save("dumbbell_classviz.png")
    print("Saved:", "dumbbell_classviz.png")
