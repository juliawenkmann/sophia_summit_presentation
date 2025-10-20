
# Steering Talk Reproductions

Code and scripts to recreate the **key graphics** from your talk, with clean, extendable building blocks.
- Slide 10 (concept algebra / word vectors)
- Slides 12–13 (top-activating images ⇒ mono/polysemanticity)
- Slides 14–15 (feature & class visualization / DeepDream-style)

> The slide numbering, topics and references follow your PDF, e.g. *Concept algebra* on **page 10**, *max-activation image grids* on **pages 12–13**, and *gradient-ascent visualizations* on **pages 14–15**.


## Quick start

```bash
# (1) Create and activate a fresh env, then install deps
python -m venv .venv && source .venv/bin/activate   # Linux/Mac
# or: py -m venv .venv && .venv\Scripts\activate  # Windows

pip install -r requirements.txt
# For GPU-specific PyTorch builds, consult pytorch.org get-started.

# (2) Verify you can see layers of the default model (GoogLeNet / Inception v1):
python scripts/list_layers.py --model googlenet --depth 3
```

### Recreate slide 10 — Concept algebra (α = 1)

Uses `gensim` word vectors. Default model is light (`glove-wiki-gigaword-100`), switch to
`word2vec-google-news-300` if you want the *classic* `king - man + woman ≈ queen` demo.

```bash
python scripts/slide10_word_arithmetic.py --model glove-wiki-gigaword-100   --pos king woman --neg man --alpha 1.0
```

### Recreate slides 12–13 — Images that maximally activate a neuron

We scan an **ImageFolder** dataset and keep the **top-k** images that maximize a chosen
layer/channel. You can point this at ImageNet (`/path/to/imagenet/val`) or a lighter substitute
(Imagenette/your own folder).

```bash
python scripts/slide12_top_activations.py   --data /path/to/imagefolder_or_imagenet/val   --layer inception4a.branch1.conv --channel 11 --k 16   --out assets/outputs/slide12_grid.png
```

*Tip:* If you use GoogLeNet, the TF paper calls layers `mixed4a`, which corresponds roughly to
`inception4a` in torchvision.

### Recreate slides 14–15 — Feature & class visualization (gradient ascent)

- **Slide 14 (unit)**: increase the activation of a specific channel in a layer.
- **Slide 15 (class)**: increase a class logit.

```bash
# Unit / channel visualization
python scripts/slide14_feature_vis.py --layer inception4a.branch1.conv --channel 11   --steps 512 --out assets/outputs/slide14_unit11.png

# Class visualization
python scripts/slide15_class_vis.py --class_idx 543   --steps 640 --out assets/outputs/slide15_class543.png
```

Parameters (`--tv`, `--l2`, `--steps`, `--lr`) are exposed to let you trade off texture vs structure.
The core routine lives in `sttalk.viz.feature_visualization` and is kept simple on purpose.


## Optional: unify “same-ish network” across examples

All image-based examples default to **GoogLeNet (Inception v1)** from `torchvision`, so you can stick
to a single, realistic network. For **concept algebra on images**, there’s also a lightweight CLIP
addon (`sttalk.clip_tools`) that lets you compose text vectors and retrieve nearest images. Example:

```bash
# Build a CLIP index over your dataset and retrieve images for "zebra + stripes - horse"
python - <<'PY'
from sttalk.clip_tools import CLIPIndex
ci = CLIPIndex().build_from_imagefolder("/path/to/imagefolder/val")
v = ci.compose_text(["zebra","stripes"], ["horse"], alpha=1.0)
paths = ci.nearest_images_to_vector(v, topk=12)
print("\n".join(paths))
PY
```


## Bonus (if time allows) — Toy superposition

A tiny autoencoder on sparse Bernoulli–Uniform data to visualize “packing” of features in 2D:

```bash
python scripts/bonus_superposition_toy.py --d 5 --m 2 --pi 0.22 --n 50000   --epochs 15 --out assets/outputs/bonus_superposition.png
```


## Repository layout

```
steering-talk-repro/
├── src/sttalk/
│   ├── __init__.py
│   ├── utils.py                # device/seed helpers, module resolver, image I/O
│   ├── activations.py          # dataset scan + top-k activations, image grids
│   ├── viz.py                  # feature & class visualization via gradient ascent
│   ├── concept_algebra.py      # word-vector arithmetic (α = 1 by default)
│   └── clip_tools.py           # optional CLIP-based concept algebra over images
├── scripts/
│   ├── list_layers.py                  # discover layer names
│   ├── slide10_word_arithmetic.py      # reproduces slide 10
│   ├── slide12_top_activations.py      # reproduces slides 12–13
│   ├── slide14_feature_vis.py          # reproduces slide 14
│   └── slide15_class_vis.py            # reproduces slide 15
├── assets/outputs/              # generated figures will appear here
├── requirements.txt
├── pyproject.toml               # allows 'pip install -e .'
├── LICENSE
└── README.md
```


## Notes & references

- Slides **10–15** in your deck motivate these reproductions: concept algebra (Mikolov et al.
  2013), mono/polysemanticity via **max-activation image grids** (Szegedy et al. 2014) and
  **feature/class visualization** à la *Inceptionism* and Distill’s *Feature Visualization* (Olah et al.).
- If you show the *smiling woman − woman + man = smiling man* example (your slide 25), that’s a
  GAN-latent arithmetic classic (Radford et al., 2015) and not implemented here; consider BigGAN or
  StyleGAN for a faithful reproduction.

Ethics & licensing: check dataset licenses (ImageNet/Imagenette), and cite the original papers when
presenting the reproduced visuals.
