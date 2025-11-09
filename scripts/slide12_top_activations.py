"""
slide12_top_activations_cli.py
--------------------------------
Clean, reliable "top activations" script with these guarantees:
  • **Exact K per channel**: every shown grid has exactly K images (no missing).
  • **No duplicates in dataset**: content-hash de-dup on collection/materialization.
  • **No within-grid duplicates**: selection enforces unique-by-hash for each channel.
  • **Pop-up grids** (plt.show) so results appear like in a notebook.
  • **Class names on tiles** (dataset label if available, else predicted ImageNet class).

Sources
  - picsum (default; robust, unique-by-hash)
  - folder (easiest for local ImageNet: --source folder --images /path/to/imagenet/val)
  - torchvision datasets (oxford_pets, caltech101, flowers102, food101, cifar100, stl10) with fallback to picsum

Quick starts
------------
# 1) Out-of-the-box (unique picsum images, pop-up grids, exact K)
python slide12_top_activations_cli.py

# 2) Local ImageNet (or any folder), ResNet-50 layer4, top-12
python slide12_top_activations_cli.py --source folder --images "/path/to/imagenet/val" \\
    --model resnet50 --layer layer4 --topk 12

# 3) TorchVision dataset (falls back to picsum if a download fails)
python slide12_top_activations_cli.py --source torchvision --tv-dataset oxford_pets --tv-split test --num-images 400
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import random
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import requests
import torch
import torch.nn as nn
import torchvision as tv
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# ---------------------------
# CLI
# ---------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Top activations (exact K per channel, no dataset duplicates, pop-up grids).")
    # Source
    p.add_argument("--source", type=str, default="picsum",
                   choices=["picsum", "folder", "torchvision"],
                   help="Where to get images from.")
    p.add_argument("--images", type=str, default=None, help="Local folder (recursive) when --source folder.")
    p.add_argument("--data-root", type=str, default="data", help="Root for downloads/materialized images.")
    p.add_argument("--num-images", type=int, default=400, help="Target number of unique images after de-dup.")
    p.add_argument("--seed", type=int, default=0, help="PRNG seed for picsum.")
    # picsum-only
    p.add_argument("--image-size", type=int, default=512, help="Picsum fetch size (square).")
    p.add_argument("--max-attempts", type=int, default=4000, help="Picsum max attempts to get unique images.")
    # TorchVision datasets
    p.add_argument("--tv-dataset", type=str, default="oxford_pets",
                   choices=["oxford_pets", "caltech101", "flowers102", "food101", "cifar100", "stl10"],
                   help="TorchVision dataset to auto-download when --source torchvision.")
    p.add_argument("--tv-split", type=str, default="test", help="Dataset split (varies by dataset).")
    # De-duplication
    p.add_argument("--dedupe-source", type=str, default="hash", choices=["hash", "path", "none"],
                   help="Global dataset de-duplication key (hash recommended).")
    # Model & layer
    p.add_argument("--model", type=str, default="googlenet", choices=["googlenet", "resnet50", "efficientnet_b0"])
    p.add_argument("--layer", type=str, default=None, help="Layer path to hook; default depends on model.")
    p.add_argument("--pool", type=str, default="mean", choices=["mean", "max"], help="Pooling over spatial dims.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    # Top-K & display
    p.add_argument("--topk", type=int, default=8, help="Top-K images per channel (EXACT for every channel).")
    p.add_argument("--show-channels", type=int, default=6, help="How many channels to display (most mono-class first).")
    p.add_argument("--viz-cols", type=int, default=4, help="Columns in grids.")
    p.add_argument("--save-grids", action="store_true", help="Also save grids under outputs/grids/.")
    p.add_argument("--save-csv", action="store_true", help="Save CSVs (topk + dominance).")
    p.add_argument("--output-dir", type=str, default="outputs")
    # Labels
    p.add_argument("--label-source", type=str, default="auto", choices=["auto", "dataset", "pred", "both"],
                   help="Which class name to show under each tile.")
    p.add_argument("--dominance-by", type=str, default="auto", choices=["auto", "dataset", "pred"],
                   help="Which labels to use for dominance calculation (mono-class ranking).")
    return p

# ---------------------------
# Utilities
# ---------------------------
def pick_device(spec: str) -> str:
    if spec == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if spec == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return spec

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

def md5_bytes(data: bytes) -> str:
    h = hashlib.md5(); h.update(data); return h.hexdigest()

def md5_file(path: str, chunk: int = 1 << 16) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

# ---------------------------
# Models
# ---------------------------
def load_torchvision_model(name: str, device: str):
    name = name.lower()
    if name == "googlenet":
        from torchvision.models import googlenet, GoogLeNet_Weights
        weights = GoogLeNet_Weights.IMAGENET1K_V1
        model = googlenet(weights=weights)
        preprocess = weights.transforms()
        categories = weights.meta.get("categories", None)
        hook_default = "inception5b"
    elif name == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        preprocess = weights.transforms()
        categories = weights.meta.get("categories", None)
        hook_default = "layer4"
    elif name == "efficientnet_b0":
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)
        preprocess = weights.transforms()
        categories = weights.meta.get("categories", None)
        hook_default = "features.6"
    else:
        raise ValueError(f"Unsupported model: {name}")
    model.eval().to(device)
    return model, preprocess, categories, hook_default

def get_module_by_path(root: nn.Module, dotted_path: str) -> nn.Module:
    mod: nn.Module = root
    for part in dotted_path.split("."):
        if part.isdigit():
            mod = list(mod.children())[int(part)]
        else:
            mod = getattr(mod, part)
    return mod

# ---------------------------
# Sources
# ---------------------------
def list_images_recursive(root: Path) -> List[str]:
    if root.is_file():
        return [str(root)]
    paths = [str(p) for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    paths.sort()
    return paths

def save_jpeg_bytes(img: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def materialize_dataset(ds, out_dir: Path, target_n: int) -> Tuple[List[str], Dict[str, str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []
    path2label: Dict[str, str] = {}
    seen_hash = set()
    names = getattr(ds, "classes", None) or getattr(ds, "categories", None)

    for i in tqdm(range(len(ds)), desc="materialize", unit="img"):
        if target_n and len(saved_paths) >= target_n:
            break
        img, target = ds[i]
        try:
            data = save_jpeg_bytes(img, quality=92)
            h = md5_bytes(data)
            if h in seen_hash:
                continue
            seen_hash.add(h)
            p = out_dir / f"{h}.jpg"
            if not p.exists():
                with open(p, "wb") as f:
                    f.write(data)
            saved_paths.append(str(p))
            if isinstance(names, (list, tuple)) and 0 <= int(target) < len(names):
                path2label[str(p)] = str(names[int(target)])
            else:
                path2label[str(p)] = str(int(target))
        except Exception:
            continue
    return saved_paths, path2label

def collect_source(args) -> Tuple[List[str], Dict[str, str]]:
    root = Path(args.data_root); root.mkdir(parents=True, exist_ok=True)

    if args.source == "picsum":
        random.seed(args.seed)
        out = root / "picsum"; out.mkdir(parents=True, exist_ok=True)
        saved, seen = [], set()
        with tqdm(total=args.num_images, desc="picsum", unit="img") as bar:
            attempts = 0
            while len(saved) < args.num_images and attempts < args.max_attempts:
                attempts += 1
                seed = random.randint(0, 10_000_000)
                url = f"https://picsum.photos/seed/{seed}/{args.image_size}/{args.image_size}"
                try:
                    r = requests.get(url, timeout=(5, 20))
                    if r.status_code == 200 and r.content:
                        h = md5_bytes(r.content)
                        if h in seen: continue
                        seen.add(h)
                        p = out / f"{h}.jpg"
                        if not p.exists():
                            with open(p, "wb") as f:
                                f.write(r.content)
                        saved.append(str(p)); bar.update(1)
                except Exception:
                    continue
        return saved, {}

    if args.source == "folder":
        if not args.images:
            raise RuntimeError("--images is required for --source folder")
        paths = list_images_recursive(Path(args.images))
        if not paths:
            raise RuntimeError(f"No images under {args.images}")
        return paths, {}

    # TorchVision (with graceful fallback to picsum)
    cache = root / "torchvision_cache"; cache.mkdir(parents=True, exist_ok=True)
    try:
        name = args.tv_dataset
        if name == "oxford_pets":
            ds = tv.datasets.OxfordIIITPet(root=str(cache), split=args.tv_split, download=True)
        elif name == "caltech101":
            ds = tv.datasets.Caltech101(root=str(cache), download=True)
        elif name == "flowers102":
            ds = tv.datasets.Flowers102(root=str(cache), split=args.tv_split, download=True)
        elif name == "food101":
            ds = tv.datasets.Food101(root=str(cache), split=args.tv_split, download=True)
        elif name == "cifar100":
            ds = tv.datasets.CIFAR100(root=str(cache), train=(args.tv_split!="test"), download=True)
        elif name == "stl10":
            ds = tv.datasets.STL10(root=str(cache), split=args.tv_split, download=True)
        else:
            raise ValueError(f"Unsupported tv-dataset: {name}")
        out = root / f"tv_{name}_{args.tv_split}"
        paths, p2l = materialize_dataset(ds, out, args.num_images)
        if not paths:
            raise RuntimeError("TorchVision dataset materialization yielded no images.")
        return paths, p2l
    except Exception as e:
        print(f"[WARN] TorchVision dataset '{args.tv_dataset}' failed ({e}). Falling back to picsum.")
        args.source = "picsum"
        return collect_source(args)

# ---------------------------
# Global dataset de-dup
# ---------------------------
def dedupe_paths(paths: List[str], mode: str) -> Tuple[List[str], List[str]]:
    if mode == "none":
        return paths, list(paths)
    if mode == "path":
        seen = set(); kept = []; keys = []
        for p in paths:
            if p in seen: continue
            seen.add(p); kept.append(p); keys.append(p)
        return kept, keys
    if mode == "hash":
        seen = set(); kept = []; keys = []
        for p in tqdm(paths, desc="dedupe(hash)", unit="img"):
            try:
                k = md5_file(p)
            except Exception:
                continue
            if k in seen: continue
            seen.add(k); kept.append(p); keys.append(k)
        return kept, keys
    raise ValueError(f"Unknown mode: {mode}")

# ---------------------------
# Activations & Top-K
# ---------------------------
def pool_activations(acts: torch.Tensor, pool: str) -> torch.Tensor:
    if acts.ndim == 4:
        if pool == "mean": return acts.mean(dim=(2,3))
        if pool == "max":  return acts.amax(dim=(2,3))
        raise ValueError(f"Unknown pool: {pool}")
    if acts.ndim == 2: return acts
    raise ValueError(f"Unexpected activation shape: {acts.shape}")

@torch.no_grad()
def collect_scores_and_logits(model: nn.Module, hook_layer: nn.Module,
                              loader: torch.utils.data.DataLoader, device: str, pool: str):
    buf: List[torch.Tensor] = []
    def _hook(_m, _in, out): buf.append(out.detach())
    h = hook_layer.register_forward_hook(_hook)

    all_s, all_p, all_l = [], [], []
    for batch, paths in loader:
        batch = batch.to(device); buf.clear()
        logits = model(batch)
        if isinstance(logits, (tuple, list)): logits = logits[0]
        a = buf.pop(); s = pool_activations(a, pool=pool)
        all_s.append(s.cpu().numpy()); all_p.extend(list(paths))
        all_l.append(logits.detach().cpu().numpy())
    h.remove()
    scores = np.concatenate(all_s, axis=0)
    logits = np.concatenate(all_l, axis=0)
    return scores, all_p, logits

def topk_unique_exact(scores_np: np.ndarray, k: int, keys_seq: Sequence[str]) -> np.ndarray:
    """
    Exact-K selection per channel with uniqueness by 'keys_seq' (e.g., content hash).
    Raises if any channel cannot supply K unique items.
    Returns: [k, C] int array of indices.
    """
    N_loc, C_loc = scores_np.shape
    if N_loc < k:
        raise RuntimeError(f"Dataset has only N={N_loc} images < topk={k}. Increase --num-images or lower --topk.")
    order = np.argsort(-scores_np, axis=0)  # [N, C] descending
    out = np.zeros((k, C_loc), dtype=int)
    for c in range(C_loc):
        seen = set(); picked = []
        for i in order[:, c]:
            key = keys_seq[int(i)]
            if key in seen: continue
            picked.append(int(i)); seen.add(key)
            if len(picked) == k: break
        if len(picked) < k:
            raise RuntimeError(f"Channel {c} yielded only {len(picked)} unique items < topk={k}. Try lowering --topk or increasing --num-images.")
        out[:, c] = picked
    return out

# ---------------------------
# Visualization
# ---------------------------
def label_for_tile(path: str, pred_name: str, path2dslabel: Dict[str, str], label_source: str) -> str:
    if label_source == "dataset":
        return path2dslabel.get(path, pred_name)
    if label_source == "pred":
        return pred_name
    if label_source == "both":
        ds = path2dslabel.get(path, "")
        if ds and ds != pred_name:
            return f"{ds} | {pred_name}"
        return ds or pred_name
    # auto
    return path2dslabel.get(path, pred_name)

def show_grid(indices: Sequence[int],
              paths: Sequence[str],
              scores: np.ndarray,
              pred_names: Sequence[str],
              path2dslabel: Dict[str, str],
              channel: int,
              title: str,
              savepath: Optional[Path],
              ncols: int = 4,
              figsize: Tuple[int, int] = (10, 8),
              label_source: str = "auto") -> None:
    n = len(indices); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1: axes = np.array([[axes]])
    elif nrows == 1: axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)

    for j, ax in enumerate(axes.ravel()):
        if j < n:
            idx = int(indices[j]); p = paths[idx]
            try:
                im = Image.open(p).convert("RGB")
            except Exception:
                ax.axis("off"); ax.set_title("Error", fontsize=9); continue
            ax.imshow(im)
            label_txt = label_for_tile(p, pred_names[idx], path2dslabel, label_source)
            if len(label_txt) > 40: label_txt = label_txt[:37] + "…"
            ax.set_title(f"#{idx}  {scores[idx, channel]:.3f}\n{label_txt}", fontsize=9)
            ax.axis("off")
        else:
            ax.axis("off")

    fig.suptitle(title, fontsize=12); fig.tight_layout()
    if savepath is not None:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()

# ---------------------------
# Main
# ---------------------------
def main():
    args = build_parser().parse_args()

    # 1) Collect and de-dup source
    paths_raw, path2dslabel = collect_source(args)
    if not paths_raw: raise RuntimeError("No images collected.")

    if args.dedupe_source != "none":
        paths, keys = dedupe_paths(paths_raw, mode=args.dedupe_source)
    else:
        paths = paths_raw; keys = list(paths)
    if not paths: raise RuntimeError("No images remain after de-duplication.")

    # Ensure at least topk items exist
    if len(paths) < args.topk:
        raise RuntimeError(f"Need at least {args.topk} unique images; got {len(paths)}. Increase --num-images or lower --topk.")

    # If too many, truncate to num-images (after dedupe)
    if args.num_images and len(paths) > args.num_images:
        paths = paths[:args.num_images]

    # 2) Model
    device = pick_device(args.device)
    model, preprocess, categories, default_layer = load_torchvision_model(args.model, device)
    layer_path = args.layer or default_layer
    try:
        hook_layer = get_module_by_path(model, layer_path)
    except Exception as e:
        raise RuntimeError(f"Could not resolve layer '{layer_path}' on model '{args.model}': {e}")

    # 3) DataLoader
    class ImageListDataset(torch.utils.data.Dataset):
        def __init__(self, paths: Sequence[str], transform: Callable):
            self.paths = list(paths); self.t = transform
        def __len__(self) -> int: return len(self.paths)
        def __getitem__(self, i: int):
            p = self.paths[i]
            with Image.open(p) as im: im = im.convert("RGB")
            return self.t(im), p

    loader = torch.utils.data.DataLoader(ImageListDataset(paths, preprocess),
                                         batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers)

    # 4) Forward
    scores, sample_paths, logits = collect_scores_and_logits(model, hook_layer, loader, device=device, pool=args.pool)
    N, C = scores.shape
    print(f"Pooled activation scores: shape = {scores.shape} (N={N}, C={C})")
    if N < args.topk:
        raise RuntimeError(f"Post-load N={N} < topk={args.topk}; lower --topk or increase images.")

    # 5) Predicted class names
    pred_ids = np.argmax(logits, axis=1)
    pred_names = [categories[i] if (categories is not None and i < len(categories)) else str(i) for i in pred_ids]

    # 6) Keys aligned to sample order
    if args.dedupe_source == "hash":
        keys_in_order = [md5_file(p) for p in sample_paths]
    elif args.dedupe_source == "path":
        keys_in_order = list(sample_paths)
    else:
        keys_in_order = [str(i) for i in range(len(sample_paths))]

    # 7) Exact-K top per channel (unique within channel)
    top_idx = topk_unique_exact(scores, k=args.topk, keys_seq=keys_in_order)  # [k, C]

    # 8) Dominance ranking (choose labels: dataset if available, else predicted, or per --dominance-by)
    if args.dominance_by == "dataset" or (args.dominance_by == "auto" and any(p in path2dslabel for p in sample_paths)):
        names_for_dom = [path2dslabel.get(p, pred_names[i]) for i, p in enumerate(sample_paths)]
    else:
        names_for_dom = pred_names

    dominance_rows = []
    for c in range(C):
        idxs = [int(top_idx[r, c]) for r in range(args.topk)]
        names = [names_for_dom[i] for i in idxs]
        counts = Counter(names)
        dom_name, dom_count = counts.most_common(1)[0] if counts else ("", 0)
        frac = dom_count / float(args.topk)
        dominance_rows.append({"channel": c, "dominant_class": dom_name,
                               "dominant_count": dom_count, "k": args.topk, "fraction": frac})
    dominance_rows.sort(key=lambda d: (-d["fraction"], -d["dominant_count"], d["channel"]))

    print("\nMost mono-class channels (K={} exact):".format(args.topk))
    for row in dominance_rows[:args.show_channels]:
        print(f"  ch {row['channel']:>4}: {row['dominant_class']}  ({row['dominant_count']}/{row['k']}; {row['fraction']:.2f})")

    # 9) CSVs (optional)
    out_dir = Path(args.output_dir)
    if args.save_csv:
        out_dir.mkdir(parents=True, exist_ok=True)
        topk_csv = out_dir / "topk_summary.csv"
        with open(topk_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["channel", "rank", "index", "path", "score", "pred_name", "dataset_name"])
            for c in range(C):
                for r in range(args.topk):
                    i = int(top_idx[r, c])
                    w.writerow([c, r, i, sample_paths[i], float(scores[i, c]), pred_names[i], path2dslabel.get(sample_paths[i], "")])
        dom_csv = out_dir / "channel_class_dominance.csv"
        with open(dom_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["channel", "dominant_class", "dominant_count", "k", "fraction"])
            w.writeheader(); w.writerows(dominance_rows)
        print(f"Saved CSVs to {out_dir}")

    # 10) Display (and optionally save) the top 'show-channels' monoclass grids, each with EXACT K
    label_src = args.label_source
    if label_src == "auto":
        label_src = "dataset" if any(p in path2dslabel for p in sample_paths) else "pred"

    if args.save_grids:
        (out_dir / "grids").mkdir(parents=True, exist_ok=True)

    for row in dominance_rows[:args.show_channels]:
        ch = row["channel"]
        idxs = [int(top_idx[r, ch]) for r in range(args.topk)]
        title = f"Channel {ch} — top {args.topk}"
        savepath = (out_dir / "grids" / f"grid_channel_{ch}.png") if args.save_grids else None
        show_grid(idxs, sample_paths, scores, pred_names, path2dslabel,
                  channel=ch, title=title, savepath=savepath,
                  ncols=args.viz_cols, label_source=label_src)

    print("\nDone.")

if __name__ == "__main__":
    main()