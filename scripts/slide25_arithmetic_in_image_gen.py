#!/usr/bin/env python3
"""
Vector arithmetic in latent space (pure Python / PyTorch, normal display).

- Loads a PyTorch DCGAN generator (auto-downloads CelebA 64x64 weights unless
  --weights is provided).
- Samples three latent batches A, B, C ~ N(0, I), computes Y = A - B + C row-wise.
- Generates images for [A | B | C | Y], arranges them into a 4-column grid.
- Shows the figure via Matplotlib and saves a PNG.
"""

import argparse
import os
from pathlib import Path
from datetime import datetime
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ------------------------------ Paths & defaults ------------------------------
SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]                    # .../ROOT
OUT_DIR = ROOT / "scripts" / "out"
WEIGHTS_DIR = ROOT / "scripts" / "weights"
OUT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Pretrained PyTorch generator (CelebA 64x64) — hosted on Hugging Face
# Model card: https://huggingface.co/hussamalafandi/DCGAN_CelebA
HF_GENERATOR_URL = (
    "https://huggingface.co/hussamalafandi/DCGAN_CelebA/resolve/main/generator.pth"
)
HF_LOCAL_WEIGHTS = WEIGHTS_DIR / "celeba_generator_hf.pth"

# Default DCGAN generator config for CelebA 64x64
DEFAULT_LATENT_DIM = 100
DEFAULT_NGF = 64
DEFAULT_NC = 3
DEFAULT_IMAGE_SIZE = 64


# ------------------------------ Model definition ------------------------------
# Matches standard DCGAN generator used for 64x64 images (ConvTranspose2d blocks)
# and is consistent with the architecture published with the HF weights.
class Generator(nn.Module):
    def __init__(self, latent_dim=100, ngf=64, nc=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.ngf = ngf
        self.nc = nc

        self.main = nn.Sequential(
            # input Z: (N, latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),                       # (ngf*8) x 4 x 4

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),                       # (ngf*4) x 8 x 8

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),                       # (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),                       # (ngf) x 32 x 32

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),                           # (nc) x 64 x 64 in [-1, 1]
        )

    def forward(self, z):
        return self.main(z)


# ------------------------------ Utilities ------------------------------
def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_celebA_weights(local_path: Path = HF_LOCAL_WEIGHTS) -> Path:
    """Download CelebA DCGAN generator weights if not present."""
    if local_path.exists():
        return local_path
    print(f"[INFO] Downloading pretrained CelebA generator -> {local_path}")
    try:
        urlretrieve(HF_GENERATOR_URL, local_path.as_posix())
    except Exception as e:
        raise RuntimeError(
            "Failed to download pretrained weights. "
            f"Try again later or provide --weights. Error: {e}"
        )
    return local_path


def to_uint8_image(t: torch.Tensor) -> np.ndarray:
    """
    Map a CHW float tensor in [0,1] to HxWxC uint8 for display/saving.
    """
    t = t.clamp(0, 1)
    arr = (t.detach().cpu().numpy() * 255.0).astype(np.uint8)
    return np.transpose(arr, (1, 2, 0))  # HWC


def stack_rows(batch: torch.Tensor) -> np.ndarray:
    """
    Vertically stack a batch of images (N, C, H, W) into a single H*N x W x C image.
    Assumes values already in [0,1].
    """
    imgs = []
    for i in range(batch.size(0)):
        imgs.append(to_uint8_image(batch[i]))
    return np.vstack(imgs)


def make_column_grid(A, B, C, Y) -> np.ndarray:
    """
    Given four batches of images (A,B,C,Y) each (N,3,64,64) in [0,1],
    produce one big tiled image with 4 columns and N rows.
    """
    colA = stack_rows(A)
    colB = stack_rows(B)
    colC = stack_rows(C)
    colY = stack_rows(Y)
    return np.hstack([colA, colB, colC, colY])  # H_total x (4*W) x C


# ------------------------------ Main procedure ------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Vector arithmetic in latent space (A - B + C) with a DCGAN generator (PyTorch)."
    )
    parser.add_argument("--rows", type=int, default=8, help="Rows per column (N)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--latent", type=int, default=DEFAULT_LATENT_DIM, help="Latent dimension (nz)")
    parser.add_argument("--ngf", type=int, default=DEFAULT_NGF, help="Generator feature size (ngf)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="cpu or cuda")
    parser.add_argument("--weights", type=str, default="", help="Path to a PyTorch generator .pth (optional)")
    parser.add_argument("--out", type=str, default="", help="Output image path (PNG); default in scripts/out/")
    parser.add_argument("--no-show", action="store_true", help="Do not pop up a Matplotlib window")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Resolve output path
    out_path = Path(args.out) if args.out else OUT_DIR / f"vector_arithmetic_{_timestamp()}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Device
    dev = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to CPU.")

    # Build generator and load weights
    G = Generator(latent_dim=args.latent, ngf=args.ngf, nc=DEFAULT_NC).to(dev).eval()
    if args.weights:
        ckpt_path = Path(args.weights).expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--weights file not found: {ckpt_path}")
    else:
        ckpt_path = ensure_celebA_weights()

    state = torch.load(ckpt_path, map_location=dev)
    # Support both raw state_dict and wrapped dicts
    if isinstance(state, dict) and all(k.startswith("main.") or k.startswith("module.") or k in {"latent_dim","ngf","nc"} for k in state.keys()):
        # Looks like a state_dict
        G.load_state_dict(state if "main.0.weight" in state or "module.main.0.weight" in state else state)
    elif isinstance(state, dict) and "state_dict" in state:
        G.load_state_dict(state["state_dict"])
    else:
        # Best effort: try loading directly
        G.load_state_dict(state)

    # Sample A,B,C in latent space and compute Y = A - B + C
    N = args.rows
    nz = args.latent
    zA = torch.randn(N, nz, 1, 1, device=dev)
    zB = torch.randn(N, nz, 1, 1, device=dev)
    zC = torch.randn(N, nz, 1, 1, device=dev)
    zY = zA - zB + zC

    # Generate images for all four columns
    with torch.no_grad():
        Z = torch.cat([zA, zB, zC, zY], dim=0)  # (4N, nz, 1, 1)
        out = G(Z).float()                      # [-1, 1]
        out = (out + 1.0) / 2.0                 # -> [0, 1]

    Aimgs = out[:N]
    Bimgs = out[N:2 * N]
    Cimgs = out[2 * N:3 * N]
    Yimgs = out[3 * N:4 * N]

    # Make a single tiled image (HxWx3 uint8)
    big = make_column_grid(Aimgs, Bimgs, Cimgs, Yimgs)

    # Save with titles using Matplotlib (normal display), and also write raw grid
    fig_h = max(4, int(0.15 * big.shape[0] / 10))  # heuristic for readable aspect
    fig = plt.figure(figsize=(12, 0.012 * big.shape[0] + 1.5))
    plt.imshow(big)
    plt.axis('off')
    # Titles above columns
    H, W, _ = big.shape
    col_w = W // 4
    titles = ["A", "B", "C", "A - B + C"]
    for i, t in enumerate(titles):
        x = (i + 0.5) * col_w
        plt.text(x, -0.02 * H, t, ha='center', va='bottom', fontsize=14, weight='bold', transform=plt.gca().transData)

    plt.tight_layout(pad=0.1)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.0, dpi=150)
    print(f"[OK] Saved vector arithmetic grid to: {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
