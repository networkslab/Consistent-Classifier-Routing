"""Train a flat router or a cascade on precomputed prompt and answer embeddings.

Pick a dataset, a model family and a router type. The script loads the prompt
embeddings, one answer-embedding file per model and the labels (see
:mod:`consistent_routing.features` for the file layout), splits them 80/10/10,
trains the classifiers and the router jointly with the hinge surrogate, and
evaluates the hard decisions on the test set with 10 bootstrap resamples.
Embeddings are never computed here: missing files raise an error naming the
expected path.

Examples::

    # default layout: embeddings/GSM8K/{prompt,labels,llama-3.2-1b,llama-3.3-70b,llama-3.1-405b}.npy
    python experiments/embedding_routing.py --dataset GSM8K --family llama --router flat

    # cascade with Qwen on CommonsenseQA, with one answer file given explicitly
    python experiments/embedding_routing.py --dataset CommonsenseQA --family qwen --router cascade \\
        --answer-emb qwen2.5-72b=/data/csqa/qwen72b_answers.npy --kappa 0 0.05 0.06

Writes ``results.json`` to ``--out``: configuration, test 0-1-c risk (error rate,
average cost), and the bootstrap statistics of every classifier.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from consistent_routing import CascadeHingeLoss, CascadeModel, FlatHingeLoss, FlatModel, cumulative_costs
from consistent_routing.features import DATASETS, FAMILIES, load_embeddings, split_indices
from consistent_routing.risk import bootstrap_report, zero_one_c_risk
from consistent_routing.train import TrainConfig, fit


def parse_answer_paths(pairs: Sequence[str]) -> Dict[str, str]:
    """Parse ``MODEL=PATH`` arguments."""
    paths = {}
    for pair in pairs:
        model, sep, path = pair.partition("=")
        if not sep or not model or not path:
            raise SystemExit(f"--answer-emb expects MODEL=PATH, got {pair!r}")
        paths[model] = path
    return paths


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True,
                        help=f"dataset name, e.g. {', '.join(DATASETS)} (sets K and the default directory)")
    parser.add_argument("--family", choices=sorted(FAMILIES), default=None,
                        help="model family preset; its models are used in cascade order")
    parser.add_argument("--models", nargs="+", default=None,
                        help="explicit model names in cascade order (cheapest first), instead of --family")
    parser.add_argument("--router", choices=["flat", "cascade"], required=True)
    parser.add_argument("--kappa", type=float, nargs="+", default=None,
                        help="per-classifier costs in cascade order (default: 0, 0.05, 0.07); "
                             "the cascade pays their cumulative sum")
    parser.add_argument("--emb-dir", type=Path, default=Path("embeddings"),
                        help="root of the default file layout <emb-dir>/<dataset>/")
    parser.add_argument("--prompt-emb", type=Path, default=None, help="prompt embeddings (N, d), overrides the default")
    parser.add_argument("--answer-emb", nargs="*", default=[], metavar="MODEL=PATH",
                        help="answer embeddings of one model (N, d), overrides the default; repeatable")
    parser.add_argument("--labels", type=Path, default=None, help="labels (N,), overrides the default")
    parser.add_argument("--num-classes", type=int, default=None, help="K, if the dataset is not a preset")
    parser.add_argument("--epochs", type=int, default=None, help="default: 100 (flat), 1000 (cascade)")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n-hidden", type=int, default=2, help="hidden layers of each classifier")
    parser.add_argument("--router-n-hidden", type=int, default=1, help="hidden layers of the router / deferral modules")
    parser.add_argument("--bootstrap", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None, help="default: results/<dataset>_<models>_<router>")
    args = parser.parse_args(argv)
    if (args.family is None) == (args.models is None):
        parser.error("give exactly one of --family or --models")
    args.models = args.models or FAMILIES[args.family]
    args.kappa = args.kappa or [0.0, 0.05, 0.07][: len(args.models)]
    if len(args.kappa) != len(args.models):
        parser.error(f"--kappa needs one cost per model ({len(args.models)}), got {len(args.kappa)}")
    args.epochs = args.epochs or (1000 if args.router == "cascade" else 100)
    name = args.family or "-".join(args.models)
    args.out = args.out or Path("results") / f"{args.dataset}_{name}_{args.router}"
    return args


def to_split(data, idx):
    """``(x, [z_1, ..., z_T], y)`` tensors for the rows ``idx``."""
    return (torch.from_numpy(data.x[idx]), [torch.from_numpy(z[idx]) for z in data.zs],
            torch.from_numpy(data.y[idx]))


def run(args: argparse.Namespace) -> Dict:
    torch.manual_seed(args.seed)
    data = load_embeddings(args.dataset, args.models, args.emb_dir, prompt_path=args.prompt_emb,
                           answer_paths=parse_answer_paths(args.answer_emb), labels_path=args.labels,
                           num_classes=args.num_classes)
    train_idx, val_idx, test_idx = split_indices(len(data), seed=args.seed)
    train, val, test = (to_split(data, idx) for idx in (train_idx, val_idx, test_idx))

    cascade = args.router == "cascade"
    model_cls, loss_cls = (CascadeModel, CascadeHingeLoss) if cascade else (FlatModel, FlatHingeLoss)
    costs = cumulative_costs(args.kappa) if cascade else torch.tensor(args.kappa)
    model = model_cls(data.x.shape[1], [z.shape[1] for z in data.zs], data.num_classes,
                      router_n_hidden=args.router_n_hidden, router_out_batch_norm=True,
                      hidden=args.hidden, n_hidden=args.n_hidden)
    config = TrainConfig(lr=args.lr, optimizer="adam", weight_decay=0.0, batch_size=args.batch_size,
                         max_epochs=args.epochs, patience=args.patience, seed=args.seed)
    history = fit(model, loss_cls(args.kappa), train, val, config)

    x, zs, y = test
    stage, pred = model.decide(x, zs)
    with torch.no_grad():
        stage_preds = model(x, zs)[0].argmax(dim=-1)
    results = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "models": data.models,
        "costs": [float(c) for c in costs],
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "test": zero_one_c_risk(pred, y, stage, costs),
        "bootstrap": bootstrap_report(stage, pred, stage_preds, y, costs, cascade,
                                      n_boot=args.bootstrap, seed=args.seed),
        "history": history,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    results = run(args)
    test = results["test"]
    frac = np.nanmedian(np.array(results["bootstrap"]["frac"]), axis=0)
    label = "deferred past" if args.router == "cascade" else "routed to"
    print(f"test risk {test['risk']:.4f} (error {test['error_rate']:.4f}, cost {test['avg_cost']:.4f})")
    for model, f in zip(results["models"], frac):
        print(f"  {model:>16s}: median fraction {label} {f:.3f}")
    print(f"results written to {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
