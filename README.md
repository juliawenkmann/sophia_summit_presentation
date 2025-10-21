
# Steering Talk Reproductions

Code and scripts to recreate the **key graphics** from your talk, with clean, extendable building blocks.
- Slide 10 (concept algebra / word vectors)
- Slides 12–13 (top-activating images ⇒ mono/polysemanticity)
- Slides 14–15 (feature & class visualization / DeepDream-style)

> The slide numbering, topics and references follow your PDF, e.g. *Concept algebra* on **page 10**, *max-activation image grids* on **pages 12–13**, and *gradient-ascent visualizations* on **pages 14–15**.


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



