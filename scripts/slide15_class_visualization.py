"""
Class visualization with Fourier parameterization (clean version)

What it does:
- Optimizes a frequency-domain tensor ("spec") so that, after inverse FFT,
  the resulting image strongly activates a chosen ImageNet class in GoogLeNet.
- Uses simple natural-image priors (TV + L2) and random-view augmentation
  for robustness.

Requirements:
  pip install torch torchvision pillow
"""

import math, random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms.functional as TF
from PIL import Image

# -------------------------------
# 0) Device & deterministic bits
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# -------------------------------
# 1) Load model & metadata
# -------------------------------
# We use GoogLeNet (a.k.a. Inception v1) pre-trained on ImageNet-1K.
WEIGHTS = models.GoogLeNet_Weights.IMAGENET1K_V1
NET = models.googlenet(weights=WEIGHTS, aux_logits=True).to(DEVICE).eval()

# Freeze model params (we optimize only the image).
for p in NET.parameters():
    p.requires_grad_(False)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE)[:, None, None]  # [3,1,1]
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE)[:, None, None]  # [3,1,1]
CATEGORIES = WEIGHTS.meta["categories"]  # length 1000, index -> class name


def class_index(name: str, categories=CATEGORIES) -> int:
    """
    Map a human string to the ImageNet class index.
    First try exact match; otherwise return the first 'contains' hit.
    """
    name_l = name.lower()
    try:
        return categories.index(name_l)
    except ValueError:
        hits = [i for i, s in enumerate(categories) if name_l in s.lower()]
        if not hits:
            raise ValueError(f"'{name}' not found in ImageNet categories.")
        return hits[0]


# -------------------------------
# 2) Fourier parameterization
# -------------------------------
def radial_frequency(h: int, w: int, device=DEVICE) -> torch.Tensor:
    """
    Build a (H, W/2+1) grid of radial frequencies used to scale the spectrum.
    """
    fy = torch.fft.fftfreq(h, d=1.0).to(device).reshape(h, 1)         # [H,1]
    fx = torch.fft.rfftfreq(w, d=1.0).to(device).reshape(1, w // 2 + 1)  # [1,W//2+1]
    return torch.sqrt(fx * fx + fy * fy).clamp(min=1e-6)  # [H, W//2+1]


def init_spectrum(h: int, w: int, device=DEVICE) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create the learnable rFFT spectrum:
      spec: [1, 3, H, W//2+1, 2]  (real/imag parts in the last dim)
      freqs: [H, W//2+1]
    """
    spec = torch.randn(1, 3, h, w // 2 + 1, 2, device=device, requires_grad=True)
    freqs = radial_frequency(h, w, device)
    return spec, freqs


def spectrum_to_image(spec: torch.Tensor,
                      freqs: torch.Tensor,
                      decay_power: float = 1.5) -> torch.Tensor:
    """
    Convert spectrum -> image via inverse FFT, with 1/f^decay scaling to bias
    towards natural-looking low-frequency content.

    Returns a tensor in [0,1]: [1, 3, H, W]
    """
    # Convert real/imag -> complex
    complex_spec = torch.view_as_complex(spec)  # [1,3,H,W//2+1]
    # Emphasize low frequencies (small f => bigger weight)
    scaled = complex_spec * (1.0 / (freqs ** decay_power))  # broadcast over H,W//2+1
    # Inverse FFT to spatial domain
    H, W_full = spec.shape[2], (spec.shape[3] - 1) * 2
    img = torch.fft.irfft2(scaled, s=(H, W_full), norm='ortho')  # [1,3,H,W]
    # Normalize contrast and squish to [0,1]
    img = img / (img.std(dim=(-2, -1), keepdim=True) + 1e-8)
    img = torch.tanh(img) * 0.5 + 0.5
    return img


# -------------------------------
# 3) Simple natural-image priors
# -------------------------------
def total_variation(x: torch.Tensor) -> torch.Tensor:
    """
    TV loss: encourages piecewise-smooth images.
    x: [B, C, H, W]
    """
    dx = x[:, :, 1:, :] - x[:, :, :-1, :]   # vertical differences
    dy = x[:, :, :, 1:] - x[:, :, :, :-1]   # horizontal differences
    return (dx.pow(2).mean() + dy.pow(2).mean())


def box_blur_3x3(x: torch.Tensor) -> torch.Tensor:
    """
    Light blur to suppress ringing. Keeps shape.
    """
    return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)


# -------------------------------
# 4) Random-view data augmentation
# -------------------------------
def random_view(x: torch.Tensor,
                out_hw=(224, 224),
                jitter_px: int = 24,
                max_rot_deg: float = 15.0,
                scale_range=(0.92, 1.08),
                enable_flip: bool = True,
                noise_std: float = 0.02) -> torch.Tensor:
    """
    Produce one randomly transformed & normalized view for classification.
    Input x is [1, 3, H, W]; returns [1, 3, 224, 224] (normalized).
    """
    _, _, H, W = x.shape

    # 1) Integer pixel jitter via roll (no interpolation)
    jx = int(np.random.randint(-jitter_px, jitter_px + 1))
    jy = int(np.random.randint(-jitter_px, jitter_px + 1))
    v = torch.roll(x, shifts=(jx, jy), dims=(2, 3))  # roll over H (dim=2) and W (dim=3)

    # 2) Small affine (rotation, scale, translation) — differentiable
    angle = float(random.uniform(-max_rot_deg, max_rot_deg))
    scale = float(random.uniform(*scale_range))
    tx = int(random.uniform(-0.06 * W, 0.06 * W))
    ty = int(random.uniform(-0.06 * H, 0.06 * H))
    v = TF.affine(v, angle=angle, translate=[tx, ty], scale=scale, shear=[0.0, 0.0])

    # 3) Random horizontal flip
    if enable_flip and random.random() < 0.5:
        v = TF.hflip(v)

    # 4) Resize to model input resolution
    v = F.interpolate(v, size=out_hw, mode='bilinear', align_corners=False)

    # 5) Add a touch of Gaussian noise, clamp to [0,1]
    if noise_std > 0:
        v = (v + noise_std * torch.randn_like(v)).clamp(0.0, 1.0)

    # 6) Normalize for ImageNet models
    v = (v - IMAGENET_MEAN) / IMAGENET_STD
    return v


# -------------------------------
# 5) Synthesis loop
# -------------------------------
@torch.no_grad()
def to_pil(x: torch.Tensor) -> Image.Image:
    """
    Convert [1,3,H,W] in [0,1] to a PIL image.
    """
    return TF.to_pil_image(x.squeeze(0).clamp(0, 1).cpu())


def synthesize_class(
    class_name: str = "dumbbell",
    *,
    steps: int = 700,
    n_views: int = 8,
    size: int = 384,
    lr: float = 0.08,
    tv_weight: float = 1e-4,
    l2_weight: float = 1e-6,
    decay_power: float = 1.5,
    blur_every: int = 60,
    print_every: int = 50
) -> Image.Image:
    """
    Optimize a Fourier spectrum to produce an image that the model thinks
    is the given ImageNet class.

    Core objective (maximize):
        class_score  - tv_weight * TV(img) - l2_weight * mean((img-0.5)^2)

    Returns: PIL.Image
    """
    target_idx = class_index(class_name)
    spec, freqs = init_spectrum(size, size, device=DEVICE)
    opt = torch.optim.Adam([spec], lr=lr)

    for t in range(steps):
        # 1) Build the current image from the spectrum
        img = spectrum_to_image(spec, freqs, decay_power=decay_power)  # [1,3,H,W]

        # Optional periodic blur for stability
        if blur_every and t > 0 and (t % blur_every == 0):
            img = box_blur_3x3(img)

        # 2) Robustness: evaluate mean class score over n random views
        # Each view: [1,3,224,224], batch becomes [n,3,224,224]
        batch = torch.cat([random_view(img) for _ in range(n_views)], dim=0)
        out = NET(batch)
        logits = out.logits if hasattr(out, "logits") else out  # handle aux-outputs
        class_score = logits[:, target_idx].mean()

        # 3) Priors
        tv = total_variation(img)
        l2 = ((img - 0.5) ** 2).mean()

        # 4) Maximize score with priors  ->  minimize negative
        objective = class_score - tv_weight * tv - l2_weight * l2
        loss = -objective

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # 5) Gentle spectrum damping to prevent runaway amplitudes
        with torch.no_grad():
            spec.mul_(0.995)

        if print_every and (t % print_every == 0 or t == steps - 1):
            print(f"[{t:04d}/{steps}] score={class_score.item():.3f}  TV={tv.item():.3e}  L2={l2.item():.3e}")

    final_img = spectrum_to_image(spec, freqs, decay_power=decay_power)
    return to_pil(final_img)


# -------------------------------
# 6) Example usage
# -------------------------------
if __name__ == "__main__":
    # Same spirit as your original example; uses a slightly bigger canvas and a lower decay.
    out = synthesize_class(
        "dumbbell",
        steps=1000,
        n_views=8,
        size=448,
        lr=0.01,
        decay_power=0.5,  # lower = more high-frequency detail allowed
        tv_weight=1e-4,
        l2_weight=1e-6,
        blur_every=60,
        print_every=50
    )
    out.save("dumbbell_classviz.png")
    print("Saved to dumbbell_classviz.png")
