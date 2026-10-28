"""Loading precomputed prompt and answer embeddings for routing experiments.

Embeddings are not computed here. They are read from ``.npy`` files:

* the prompt embeddings ``x``, shape ``(N, d_x)``;
* one answer-embedding file per model, ``z_t`` of shape ``(N, d_t)``, row ``i`` being the
  embedding of model ``t``'s answer to prompt ``i``;
* the labels ``y``, shape ``(N,)``, integers in ``{0, ..., K-1}``.

By default the files are looked up in ``<emb_dir>/<dataset>/`` as ``prompt.npy``,
``labels.npy`` and ``<model>.npy``, where ``<model>`` comes from the model family
presets below. Any file can be overridden with an explicit path.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

PathLike = Union[str, Path]

# Number of answer choices K per dataset.
DATASETS: Dict[str, int] = {"GSM8K": 4, "CommonsenseQA": 5}

# Models of each family, from the cheapest to the most expensive (cascade order).
FAMILIES: Dict[str, List[str]] = {
    "llama": ["llama-3.2-1b", "llama-3.3-70b", "llama-3.1-405b"],
    "qwen": ["qwen2.5-1.5b", "qwen2.5-32b", "qwen2.5-72b"],
}


@dataclass
class EmbeddingData:
    """Prompt embeddings ``x``, answer embeddings ``z_1, ..., z_T`` and labels ``y``."""

    x: np.ndarray
    zs: List[np.ndarray]
    y: np.ndarray
    models: List[str]
    num_classes: int

    def __len__(self) -> int:
        return len(self.y)


def _load_array(path: Path, what: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found at {path}. Embeddings are not computed by this code: put the file "
            "there, or pass its location with --prompt-emb, --answer-emb MODEL=PATH or --labels."
        )
    return np.load(path)


def load_embeddings(dataset: str, models: Sequence[str], emb_dir: PathLike = "embeddings",
                    prompt_path: Optional[PathLike] = None,
                    answer_paths: Optional[Mapping[str, PathLike]] = None,
                    labels_path: Optional[PathLike] = None,
                    num_classes: Optional[int] = None) -> EmbeddingData:
    """Load the embeddings and labels of ``dataset`` for ``models``, in the given order.

    Args:
        dataset: dataset name, used for the default directory and, if listed in
            :data:`DATASETS`, for the number of classes.
        models: model names in cascade order (cheapest first).
        emb_dir: root directory of the default file layout.
        prompt_path, answer_paths, labels_path: explicit files that override the
            defaults. ``answer_paths`` maps a model name to its file.
        num_classes: number of classes ``K``; inferred from :data:`DATASETS` or the labels.
    """
    root = Path(emb_dir) / dataset
    answer_paths = dict(answer_paths or {})
    unknown = set(answer_paths) - set(models)
    if unknown:
        raise ValueError(f"--answer-emb given for {sorted(unknown)}, which are not among the models {list(models)}")

    x = _load_array(Path(prompt_path) if prompt_path else root / "prompt.npy", "Prompt embeddings")
    y = _load_array(Path(labels_path) if labels_path else root / "labels.npy", "Labels").astype(np.int64)
    zs = [
        _load_array(Path(answer_paths[m]) if m in answer_paths else root / f"{m}.npy", f"Answer embeddings of {m}")
        for m in models
    ]

    sizes = {"prompt": len(x), "labels": len(y), **{m: len(z) for m, z in zip(models, zs)}}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"all files must have the same number of rows, got {sizes}")
    if x.ndim != 2 or any(z.ndim != 2 for z in zs):
        raise ValueError("embedding files must be 2-D arrays of shape (N, d)")
    k = num_classes or DATASETS.get(dataset) or int(y.max()) + 1
    if y.min() < 0 or y.max() >= k:
        raise ValueError(f"labels must lie in [0, {k - 1}], got range [{y.min()}, {y.max()}]")
    return EmbeddingData(x=x.astype(np.float32), zs=[z.astype(np.float32) for z in zs], y=y,
                         models=list(models), num_classes=k)


def split_indices(n: int, seed: int = 42, train_frac: float = 0.8, val_frac: float = 0.1):
    """Random 80/10/10 train, validation and test indices."""
    order = np.random.RandomState(seed).permutation(n)
    n_train, n_val = int(train_frac * n), int(val_frac * n)
    return order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]
