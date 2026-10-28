"""Empirical 0-1-c risk of hard multi-model decisions (eq:multi_01c).

A sample answered by classifier ``t`` pays ``1{prediction != y} + c_t``. For a
flat router ``c = kappa``. For a cascade ``c`` is cumulative
(:func:`consistent_routing.losses.cumulative_costs`).
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

from .losses import CostLike


def zero_one_c_risk(pred: Tensor, y: Tensor, stage: Tensor, costs: CostLike) -> Dict[str, float]:
    """Error rate, average cost and their sum, the empirical 0-1-c risk.

    Args:
        pred: ``(N,)`` predicted classes.
        y: ``(N,)`` labels.
        stage: ``(N,)`` index of the classifier that produced each prediction.
        costs: length-``T`` cost vector ``c``.
    """
    costs = torch.as_tensor(costs, dtype=torch.float32, device=stage.device)
    error_rate = (pred != y).float().mean().item()
    avg_cost = costs[stage].mean().item()
    return {"error_rate": error_rate, "avg_cost": avg_cost, "risk": error_rate + avg_cost}


def bootstrap_report(stage: Tensor, pred: Tensor, stage_preds: Tensor, y: Tensor, costs: CostLike,
                     cascade: bool, n_boot: int = 10, seed: int = 0) -> Dict[str, list]:
    """Per-classifier statistics over bootstrap resamples of the test set.

    For every resample (drawn with replacement, same size as the test set) and every
    classifier ``t`` it records:

    * ``all_acc``: accuracy of classifier ``t`` on all samples;
    * ``acc``: its accuracy on the samples routed to it (flat) or exiting at it
      (cascade), NaN when there are none;
    * ``frac``: fraction routed to ``t`` (flat) or deferred past ``t`` (cascade).

    It also records the 0-1-c risk of each resample under ``costs``.

    Args:
        stage: ``(N,)`` classifier chosen (flat) or exit stage (cascade).
        pred: ``(N,)`` final predictions.
        stage_preds: ``(N, T)`` predictions of every classifier on every sample.
        y: ``(N,)`` labels.
        costs: cost vector ``c`` (cumulative for a cascade).
    """
    rng = torch.Generator().manual_seed(seed)
    n, num_stages = stage_preds.shape
    report: Dict[str, list] = {"all_acc": [], "acc": [], "frac": [], "risk": []}
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=rng)
        s, p, sp, yy = stage[idx], pred[idx], stage_preds[idx], y[idx]
        correct = (sp == yy[:, None]).float()
        report["all_acc"].append(correct.mean(dim=0).tolist())
        report["acc"].append([correct[s == t, t].mean().item() if (s == t).any() else float("nan")
                              for t in range(num_stages)])
        report["frac"].append([(s > t).float().mean().item() if cascade else (s == t).float().mean().item()
                               for t in range(num_stages)])
        report["risk"].append(zero_one_c_risk(p, yy, s, costs)["risk"])
    return report
