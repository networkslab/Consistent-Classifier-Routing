"""Synthetic routing experiment (paper Fig. 2, ``fig:task``).

Trains a flat router and a cascade router on the synthetic task with
``kappa = [0, 0.05, 0.05]``, so ``c = [0, 0.05, 0.05]`` for the flat router and
``c = [0, 0.05, 0.1]`` for the cascade. It then compares the learned decisions
with the Bayes-optimal ``r*`` and ``d_1*``. Outputs written to ``--out``:

* ``results.json``: test 0-1-c risk, Bayes risk, agreement with the Bayes
  decisions on the test points, training history and configuration.
* ``flat_routing.{pdf,png}`` and ``cascade_routing.{pdf,png}``: the two panels of
  Fig. 2. The top shows the Monte-Carlo conditional risks ``R_t(x; f_t*)``,
  shaded by the Bayes-optimal decision. The bottom shows test points colored
  by the learned routing probabilities.

Usage::

    python experiments/synthetic_routing.py            # paper settings
    python experiments/synthetic_routing.py --quick    # small run, about a minute
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from consistent_routing import (  # noqa: E402
    CascadeHingeLoss,
    CascadeModel,
    FlatHingeLoss,
    FlatModel,
    cumulative_costs,
    synthetic,
    zero_one_c_risk,
)
from consistent_routing.train import TrainConfig, fit  # noqa: E402

COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c")  # f_1, f_2, f_3


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kappa", type=float, nargs=3, default=[0.0, 0.05, 0.05],
                        help="per-classifier costs kappa_1, kappa_2, kappa_3")
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--n-val", type=int, default=1_000)
    parser.add_argument("--n-test", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n-hidden", type=int, default=3)
    parser.add_argument("--n-mc", type=int, default=5_000,
                        help="Monte-Carlo draws of each of Z_2 and Z_3 for the posteriors")
    parser.add_argument("--grid", type=int, default=201, help="number of x values for the Bayes quantities")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/synthetic"))
    parser.add_argument("--quick", action="store_true", help="small sizes for a smoke test")
    args = parser.parse_args(argv)
    if args.quick:
        args.n_train, args.n_val, args.n_test = 4_000, 500, 1_000
        args.epochs, args.n_mc, args.grid = 3, 200, 101
    return args


def to_tensors(data: Dict[str, np.ndarray]):
    """``(x, [z_1, z_2, z_3], y)`` with ``z_1`` empty, as the models expect."""
    def column(values):
        return torch.as_tensor(values, dtype=torch.float32).unsqueeze(1)
    return column(data["x"]), [None, column(data["z2"]), column(data["z3"])], torch.as_tensor(data["y"])


def shade_regions(ax, grid: np.ndarray, labels: np.ndarray, colors: Sequence[str]) -> None:
    """Shade each maximal run of equal ``labels`` along ``grid`` with its color."""
    edges = np.concatenate([[grid[0]], (grid[1:] + grid[:-1]) / 2, [grid[-1]]])
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            ax.axvspan(edges[start], edges[i], color=colors[labels[start]], alpha=0.2, lw=0)
            start = i


def plot_routing(path: Path, grid: np.ndarray, risks: np.ndarray, regions: np.ndarray,
                 test_x: np.ndarray, soft: np.ndarray, cascade: bool, seed: int) -> None:
    """Draw one panel of Fig. 2 and save it as PDF and PNG."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 4), gridspec_kw={"height_ratios": [2, 1]})
    for t in range(synthetic.NUM_STAGES):
        top.plot(grid, risks[:, t], color=COLORS[t], lw=2, label=rf"$\hat{{R}}_{t + 1}(x; f^*_{t + 1})$")
    shade_regions(top, grid, regions, COLORS)
    top.set_title(r"Cascade (Bayes-optimal $d^*_1$)" if cascade else r"Flat (Bayes-optimal $r^*$)")
    top.legend(loc="upper left")

    if cascade:  # blend "exit at f_1" (blue) and "defer" (orange) by d_1
        weights = np.stack([1.0 - soft[:, 0], soft[:, 0]], axis=1)
        palette, names = COLORS[:2], [r"$\tilde{d}_1$ chose $f_1$", r"$\tilde{d}_1$ chose to defer"]
    else:  # blend the three classifier colors by the router probabilities
        weights = soft
        palette = COLORS
        names = [rf"$\tilde{{r}}$ chose $f_{t + 1}$" for t in range(synthetic.NUM_STAGES)]
    # Weighted sum of the palette colors (written out to avoid spurious BLAS warnings on macOS).
    rgb = (weights[:, :, None] * np.array([to_rgb(c) for c in palette])[None]).sum(axis=1)
    jitter = np.random.default_rng(seed).random(len(test_x))
    bottom.scatter(test_x, jitter, c=np.clip(rgb, 0, 1), s=8, alpha=0.8, linewidths=0)
    handles = [Line2D([], [], marker="o", ls="", color=c, label=n) for c, n in zip(palette, names)]
    bottom.legend(handles=handles, loc="center right")
    bottom.set_title(r"Learned $\tilde{d}_1$" if cascade else r"Learned $\tilde{r}$")
    bottom.set_xlabel(r"$x$")

    for ax in (top, bottom):
        ax.set_xlim(-1, 1)
        ax.set_yticks([])
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> Dict:
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    train, val, test = (synthetic.sample(n, rng) for n in (args.n_train, args.n_val, args.n_test))

    grid = np.linspace(-1.0, 1.0, args.grid)
    post = synthetic.monte_carlo_posteriors(grid, args.n_mc, np.random.default_rng(args.seed + 1))
    bayes = {"flat": synthetic.bayes_flat(post, args.kappa),
             "cascade": synthetic.bayes_cascade(post, args.kappa)}

    config = TrainConfig(lr=args.lr, batch_size=args.batch_size, max_epochs=args.epochs,
                         patience=args.patience, seed=args.seed)
    test_x, test_zs, test_y = to_tensors(test)
    grid_idx = synthetic.nearest_grid_index(grid, test["x"])
    args.out.mkdir(parents=True, exist_ok=True)
    results = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}

    setups = {
        "flat": (FlatModel, FlatHingeLoss, torch.tensor(args.kappa)),
        "cascade": (CascadeModel, CascadeHingeLoss, cumulative_costs(args.kappa)),
    }
    for name, (model_cls, loss_cls, costs) in setups.items():
        model = model_cls(1, synthetic.Z_DIMS, synthetic.NUM_CLASSES, hidden=args.hidden, n_hidden=args.n_hidden)
        history = fit(model, loss_cls(args.kappa), to_tensors(train), to_tensors(val), config)
        stage, pred = model.decide(test_x, test_zs)
        with torch.no_grad():
            soft = model(test_x, test_zs)[1].numpy()
        if name == "flat":
            learned, optimal, regions = stage.numpy(), bayes[name]["router"][grid_idx], bayes[name]["router"]
        else:  # stage-1 decision: defer when the sample does not exit at f_1
            learned, optimal, regions = (stage > 0).long().numpy(), bayes[name]["defer"][grid_idx], bayes[name]["defer"]
        results[name] = {
            "test": zero_one_c_risk(pred, test_y, stage, costs),
            "bayes_risk": bayes[name]["bayes_risk"],
            "agreement_with_bayes": float((learned == optimal).mean()),
            "costs": [float(c) for c in costs],
            "history": history,
        }
        plot_routing(args.out / f"{name}_routing", grid, bayes[name]["risks"], regions,
                     test["x"], soft, cascade=(name == "cascade"), seed=args.seed)

    with open(args.out / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    results = run(parse_args(argv))
    for name in ("flat", "cascade"):
        r = results[name]
        print(f"{name:8s} test risk {r['test']['risk']:.4f} (error {r['test']['error_rate']:.4f}, "
              f"cost {r['test']['avg_cost']:.4f}) | Bayes risk {r['bayes_risk']:.4f} | "
              f"agreement with Bayes decision {r['agreement_with_bayes']:.3f}")


if __name__ == "__main__":
    main()
