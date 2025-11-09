#!/usr/bin/env python3

"""
Classic Lucent feature visualization (argparse + sane defaults).
- Matches notebook usage: pass size via param.image()/param.fourier(), not render_vis kwargs.
- Displays images and saves them by default, so it "just works" from the CLI.

Install (if needed):
    pip install lucent torch torchvision pillow matplotlib

Usage (defaults mirror the notebook):
    python slide14_feature_visualization_fixed.py

Options:
    --layer, --unit, --goal, --img-size, --pixel-init, --no-show, --no-save
"""
import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch

def _add_lucent_path():
    # 1) If user set LUCENT_ROOT, prefer that
    env = os.getenv("LUCENT_ROOT")
    if env and Path(env).exists():
        sys.path.insert(0, str(Path(env).resolve()))
        print(f"[lucent] using LUCENT_ROOT: {Path(env).resolve()}")
        return

    # 2) Try common repo-relative locations based on this file's path
    here = Path(__file__).resolve()
    root = here.parent.parent  # ROOT/ (scripts/ is one level below)
    candidates = [
        root / "lucent",
        root / "src" / "lucent",
        here.parent / "src" / "lucent",  
    ]
    for c in candidates:
        if c.exists():
            sys.path.insert(0, str(c.resolve()))
            print(f"[lucent] added to sys.path: {c.resolve()}")
            return

_add_lucent_path()



from lucent.optvis import render
from lucent.optvis import objectives, param, transform  # noqa: F401
from lucent.modelzoo import inceptionv1



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = inceptionv1.inceptionv1(pretrained=True)
#model = resnet50(pretrained=True)
#model = googlenet(pretrained=True)
_ = model.to(device).eval()

goal = "mixed4a:11"
_ = render.render_vis(model, goal, show_inline=True, save_image=f"Visualization_{goal}.pdf")



obj = "mixed4a:11"

pdf_out = "/Users/juliawenkmann/Documents/CodingProjects/damien/sophia_summit/sophia_summit_presentation/figures/feature_progression_mixed4a_11random_in_fourier.pdf"

imgs = render.render_feature_with_snapshots(
    model,
    obj,
    snapshot_steps=[0, 4, 48, 2048],
    img_size=224,
    device="auto",  # cuda > mps > cpu
    pdf_path=pdf_out,
    left_text="Starting from random noise in fourier space, we optimize an image to activate a particular neuron (layer mixed4a, unit 11).",
)
print("Saved PDF:", pdf_out)


pdf_out = "/Users/juliawenkmann/Documents/CodingProjects/damien/sophia_summit/sophia_summit_presentation/figures/feature_progression_mixed4a_11_random_in_pixel.pdf"

imgs = render.render_feature_with_snapshots(
    model,
    obj,
    snapshot_steps=[0, 4, 48, 2048],
    img_size=224,
    device="auto",  # cuda > mps > cpu
    pdf_path=pdf_out,
    left_text="Starting from random noise in pixel space, we optimize an image to activate a particular neuron (layer mixed4a, unit 11).",
    fft = False
)
print("Saved PDF:", pdf_out)