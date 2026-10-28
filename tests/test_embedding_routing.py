import json
import sys
from pathlib import Path

import numpy as np
import pytest

from consistent_routing.features import FAMILIES, load_embeddings, split_indices

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import embedding_routing  # noqa: E402


def _write_dataset(root, dataset, models, n=300, dim=8, num_classes=4, seed=0):
    rng = np.random.default_rng(seed)
    folder = root / dataset
    folder.mkdir(parents=True)
    np.save(folder / "prompt.npy", rng.normal(size=(n, dim)).astype(np.float32))
    np.save(folder / "labels.npy", rng.integers(0, num_classes, n))
    for m in models:
        np.save(folder / f"{m}.npy", rng.normal(size=(n, dim)).astype(np.float32))
    return folder


def test_load_embeddings_default_layout_and_override(tmp_path):
    models = FAMILIES["llama"]
    folder = _write_dataset(tmp_path, "GSM8K", models)
    other = tmp_path / "other.npy"
    np.save(other, np.ones((300, 5), dtype=np.float32))
    data = load_embeddings("GSM8K", models, tmp_path, answer_paths={models[2]: other})
    assert data.num_classes == 4
    assert data.x.shape == (300, 8)
    assert [z.shape[1] for z in data.zs] == [8, 8, 5]
    assert np.array_equal(data.zs[2], np.ones((300, 5)))
    assert len(data) == 300 and data.models == models
    assert (folder / "prompt.npy").exists()


def test_missing_file_names_the_expected_path(tmp_path):
    _write_dataset(tmp_path, "GSM8K", FAMILIES["llama"][:2])
    with pytest.raises(FileNotFoundError, match="llama-3.1-405b"):
        load_embeddings("GSM8K", FAMILIES["llama"], tmp_path)


def test_mismatched_rows_are_rejected(tmp_path):
    models = FAMILIES["qwen"]
    folder = _write_dataset(tmp_path, "CommonsenseQA", models, num_classes=5)
    np.save(folder / f"{models[1]}.npy", np.zeros((10, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="same number of rows"):
        load_embeddings("CommonsenseQA", models, tmp_path)


def test_split_is_80_10_10_and_disjoint():
    train, val, test = split_indices(1221, seed=42)
    assert (len(train), len(val), len(test)) == (976, 122, 123)
    assert len(set(train) | set(val) | set(test)) == 1221


@pytest.mark.parametrize("router", ["flat", "cascade"])
def test_runner_smoke(tmp_path, router):
    _write_dataset(tmp_path, "GSM8K", FAMILIES["llama"])
    out = tmp_path / "out"
    args = embedding_routing.parse_args([
        "--dataset", "GSM8K", "--family", "llama", "--router", router, "--emb-dir", str(tmp_path),
        "--kappa", "0", "0.05", "0.07", "--epochs", "2", "--hidden", "16", "--bootstrap", "3", "--out", str(out),
    ])
    results = embedding_routing.run(args)
    expected_costs = [0.0, 0.05, 0.12] if router == "cascade" else [0.0, 0.05, 0.07]
    assert results["costs"] == pytest.approx(expected_costs)
    assert results["split_sizes"] == {"train": 240, "val": 30, "test": 30}
    assert len(results["bootstrap"]["frac"]) == 3
    if router == "cascade":
        assert all(row[2] == 0.0 for row in results["bootstrap"]["frac"])  # nothing is deferred past the last stage
    else:
        assert all(sum(row) == pytest.approx(1.0) for row in results["bootstrap"]["frac"])
    assert json.loads((out / "results.json").read_text())["models"] == FAMILIES["llama"]
