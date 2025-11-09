"""
Concept algebra" with CLIP" (and optional Word2Vec).

What it does
------------
- Loads a CLIP text model (e.g., ViT-B/32).
- Computes text embeddings for a small set of words either from the final layer
  or an intermediate transformer layer.
- Forms difference vectors for analogy-style pairs (a - b).
- Shows a cosine-similarity matrix over the difference vectors.
- Projects word embeddings to 2D with PCA and plots them (arrows from b -> a).
- Optionally computes the same using Word2Vec if available.

Usage
-----
python slide10_concept_algebra.py \
    --model "ViT-B/32" \
    --layer final           # or an integer layer index, e.g., 0, 1, ...
    --device auto           # "auto", "cuda", or "cpu"
    --prompt-template "{}"  # format string around each word
    --w2v-path /path/to/GoogleNews-vectors-negative300.bin   # optional
    --skip-w2v             # skip Word2Vec step (default if gensim not present)
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

# --- CLIP (pip package: clip-anytorch) ---
import clip  # noqa: F401

# --- Word2Vec (optional) ---
try:
    import gensim.downloader as api  # type: ignore
    from gensim.models import KeyedVectors  # type: ignore
except Exception:
    api = None  # type: ignore
    KeyedVectors = None  # type: ignore

# ---------------------------
# Data configuration
# ---------------------------
PAIR_STRINGS: List[Tuple[str, str]] = [
    ("queen", "king"),
    ("woman", "man"),
    ("aunt", "uncle"),
]

# Encode the raw word by default; you can change this to e.g. "a photo of a {}"
DEFAULT_PROMPT_TEMPLATE: str = "{}"

# Derived
WORDS: List[str] = sorted({w for pair in PAIR_STRINGS for w in pair})

# ---------------------------
# Helpers & types
# ---------------------------
@dataclass
class EmbeddingResult:
    vectors: Dict[str, np.ndarray]           # word -> vector
    diffs: Dict[str, np.ndarray]             # "a-b" -> difference vector

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between 1D vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def pair_similarity_matrix(diffs: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Cosine similarity matrix across difference vectors."""
    keys = list(diffs.keys())
    M = np.zeros((len(keys), len(keys)), dtype=np.float32)
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            M[i, j] = cosine_sim(diffs[ki], diffs[kj])
    return pd.DataFrame(M, index=keys, columns=keys)

# ---------------------------
# CLIP utilities
# ---------------------------
def pick_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name

def load_clip_model(name: str, device: str):
    """Load CLIP model by name; returns the model in eval mode."""
    model, _ = clip.load(name, device=device, jit=False)
    model.eval()
    return model

def encode_clip_text(model, texts: List[str], layer_idx: Optional[int] = None, device: str = "cpu") -> np.ndarray:
    """
    Encode a list of texts using CLIP's text transformer.
    If layer_idx is None, returns the final-layer embeddings.
    Otherwise, runs up to (and including) the specified residual block (0-indexed) and applies the final projection.
    """
    tokens = clip.tokenize(texts, context_length=model.context_length).to(device)

    with torch.no_grad():
        x = model.token_embedding(tokens).type(model.dtype)  # [batch, n_ctx, width]
        x = x + model.positional_embedding.type(model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        if layer_idx is None:
            # Full forward through all blocks
            x = model.transformer(x)  # uses internal causal mask
        else:
            # Manual forward to stop after a chosen block
            attn_mask = getattr(model.transformer, "attn_mask", None)
            for i, blk in enumerate(model.transformer.resblocks):
                x = blk(x, attn_mask=attn_mask)
                if i == layer_idx:
                    break

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = model.ln_final(x).type(model.dtype)

        # Take features at the [EOT] token (max token index). Matches CLIP's encode_text pooling.
        eot = tokens.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eot] @ model.text_projection

        # Normalize for cosine geometry
        x = x.float()
        x = x / x.norm(dim=-1, keepdim=True)

    return x.cpu().numpy()

def collect_clip_embeddings(words: List[str], model, layer_idx: Optional[int], prompt_template: str, device: str) -> EmbeddingResult:
    texts = [prompt_template.format(w) for w in words]
    vecs = encode_clip_text(model, texts, layer_idx=layer_idx, device=device)
    mapping = {w: vecs[i] for i, w in enumerate(words)}
    diffs = {f"{a}-{b}": mapping[a] - mapping[b] for a, b in PAIR_STRINGS}
    return EmbeddingResult(mapping, diffs)

# ---------------------------
# Word2Vec utilities (optional)
# ---------------------------
def load_word2vec(local_path: Optional[str], key: str):
    """Load Word2Vec either from a local .bin file or via gensim downloader."""
    if local_path and os.path.exists(local_path):
        if KeyedVectors is None:
            raise RuntimeError("gensim is not available but a local path was provided.")
        return KeyedVectors.load_word2vec_format(local_path, binary=True)
    if api is None:
        raise RuntimeError("gensim downloader is not available and no local path is set.")
    return api.load(key)  # requires internet the first time

def collect_w2v_embeddings(words: List[str], kv) -> EmbeddingResult:
    mapping: Dict[str, np.ndarray] = {}
    for w in words:
        try:
            mapping[w] = kv[w]
        except KeyError:
            # Skip words missing in the model's vocabulary
            pass
    diffs = {f"{a}-{b}": mapping[a] - mapping[b] for a, b in PAIR_STRINGS if a in mapping and b in mapping}
    return EmbeddingResult(mapping, diffs)

# ---------------------------
# PCA & plotting
# ---------------------------
def pca_coords(vectors: Dict[str, np.ndarray], n_components: int = 2) -> Dict[str, np.ndarray]:
    keys = list(vectors.keys())
    X = np.stack([vectors[k] for k in keys], axis=0)
    comps = PCA(n_components=n_components, random_state=0).fit_transform(X)
    return {k: comps[i] for i, k in enumerate(keys)}

def plot_pca(words: List[str], emb: EmbeddingResult, title: str = "", savepath: Optional[str] = None) -> None:
    """
    Plots words in 2D (PCA) and draws headless arrows from each source (b) to destination (a)
    for every (a, b) in PAIR_STRINGS.

    Styling:
      - source nodes (b): red
      - destination nodes (a): orange
      - arrows: blue (no arrowheads)
    """
    if not emb.vectors:
        print("Nothing to plot.")
        return

    # 2D coordinates for all words we have vectors for
    coords = pca_coords(emb.vectors, n_components=2)

    # Partition nodes into sources (b) and destinations (a)
    sources = {b for (a, b) in PAIR_STRINGS}
    destinations = {a for (a, b) in PAIR_STRINGS}

    # Keep only words that are present in coords
    src_list = [w for w in words if (w in sources and w in coords)]
    dst_list = [w for w in words if (w in destinations and w in coords)]

    # NOTE: if a word appears in both roles, we give "source" (red) priority by plotting it last.

    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))

    # --- Edges: draw blue headless "arrows" (just line segments) from b -> a
    for a, b in PAIR_STRINGS:
        if a in coords and b in coords:
            x0, y0 = coords[b]
            x1, y1 = coords[a]
            plt.plot([x0, x1], [y0, y1], linewidth=2.0, color="blue", alpha=0.85, zorder=1)

    # --- Destination nodes: ORANGE
    if dst_list:
        xd = [coords[w][0] for w in dst_list]
        yd = [coords[w][1] for w in dst_list]
        plt.scatter(xd, yd, s=70, color="orange", edgecolors="none", label="destination", zorder=2)
        for w in dst_list:
            x, y = coords[w]
            plt.text(x, y, w, ha="center", va="bottom", fontsize=10, zorder=3)

    # --- Source nodes: RED (plotted last so they appear on top if a word is in both sets)
    if src_list:
        xs = [coords[w][0] for w in src_list]
        ys = [coords[w][1] for w in src_list]
        plt.scatter(xs, ys, s=70, color="red", edgecolors="none", label="source", zorder=4)
        for w in src_list:
            x, y = coords[w]
            plt.text(x, y, w, ha="center", va="bottom", fontsize=10, zorder=5)

    # --- Aesthetics
    plt.title(title or "PCA projection")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.axis("equal")
    plt.legend(frameon=False)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------
# Main
# ---------------------------
def run(
    model_name: str,
    layer_spec: str,
    device_spec: str,
    prompt_template: str,
    use_w2v: bool,
    w2v_path: Optional[str],
    w2v_key: str = "word2vec-google-news-300",
) -> None:
    device = pick_device(device_spec)
    model = load_clip_model(model_name, device)
    num_layers = len(list(model.transformer.resblocks))
    print(f"Loaded CLIP: {model_name} | Transformer layers: {num_layers} | Device: {device}")

    # Parse layer
    layer_index: Optional[int]
    if layer_spec.lower() == "final":
        layer_index = None
    else:
        layer_index = int(layer_spec)

    # CLIP
    clip_emb = collect_clip_embeddings(WORDS, model, layer_idx=layer_index, prompt_template=prompt_template, device=device)
    title = f"CLIP PCA — layer {'final' if layer_index is None else layer_index}"
    plot_pca(WORDS, clip_emb, title=title)
    print("\n" + title + " | diff-vector cosine(sim):")
    print(pair_similarity_matrix(clip_emb.diffs).round(3))

    # Word2Vec (optional)
    if use_w2v:
        try:
            kv = load_word2vec(w2v_path, w2v_key)
        except Exception as e:
            print(f"Skipping Word2Vec: {e}")
            kv = None

        if kv is not None:
            w2v_emb = collect_w2v_embeddings(WORDS, kv)
            title2 = "Word2Vec PCA"
            plot_pca(WORDS, w2v_emb, title=title2)
            print("\n" + title2 + " | diff-vector cosine(sim):")
            print(pair_similarity_matrix(w2v_emb.diffs).round(3))

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Concept algebra with CLIP (and optional Word2Vec).")
    p.add_argument("--model", type=str, default="ViT-B/32", help="CLIP model name (e.g., ViT-B/32, ViT-L/14).")
    p.add_argument("--layer", type=str, default="final", help='"final" or an integer residual block index (0-based).')
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help='Device to run on.')
    p.add_argument("--prompt-template", type=str, default=DEFAULT_PROMPT_TEMPLATE, help='Format string for each word, e.g., "a photo of a {}".')
    p.add_argument("--w2v-path", type=str, default=None, help="Optional path to a local word2vec .bin file.")
    p.add_argument("--skip-w2v", action="store_true", help="Skip Word2Vec step (default if gensim not available).")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    use_w2v = (KeyedVectors is not None or api is not None) and not args.skip_w2v
    run(
        model_name=args.model,
        layer_spec=args.layer,
        device_spec=args.device,
        prompt_template=args.prompt_template,
        use_w2v=use_w2v,
        w2v_path=args.w2v_path,
    )
