"""Synthetic task with known conditional distributions (paper, synthetic experiments).

There are ``T = 3`` classifiers and ``K = 3`` classes. ``X``, ``Z_2`` and ``Z_3``
are independent ``Uniform[-1, 1]`` variables, ``Z_1`` is empty, and

    Y | X, Z_2, Z_3 ~ Categorical(theta(x, z_2, z_3)),   theta = softmax(l),
    l_1 = 1.2 - 4|x|
    l_2 = -0.2 + 2.2 x + 3 g(x) z_2
    l_3 = -0.2 - 2.2 x + 3 g(x) z_3
    g(x) = sigmoid(10 (|x| - 0.15)).

Classifier ``f_1`` sees ``x``, ``f_2`` sees ``(x, z_2)`` and ``f_3`` sees ``(x, z_3)``.

The Bayes-optimal decisions follow the paper's lemmas. Write
``R_t(x; f_t*) = 1 - E[max_y p(y | x, Z_t) | X = x] + c_t`` for the conditional risk
of classifier ``t``, with ``f_t*(x, z_t) = argmax_y p(y | x, z_t)``.

* Flat: ``r*(x) = argmin_t R_t(x; f_t*)``, with ``c = kappa``.
* Cascade: ``d_1*(x) = 1{R_1(x) > E[V_2(x, Z_2) | X = x]}``, with cumulative costs.
  Here ``V_2(x, z_2) = min(1 - max_y p(y | x, z_2) + c_2, R_3(x, z_2))``, and
  ``R_3(x, z_2)`` is the expected risk of continuing to ``f_3*``.

The posteriors ``p(y | x)``, ``p(y | x, z_2)`` and ``p(y | x, z_3)`` are Monte-Carlo
estimates.

The cascade value ``V_2`` uses the stage-3 risk given the history ``(x, z_2)``,
``R_3(x, z_2) = 1 - E[theta_{f_3*(x, Z_3)}(x, z_2, Z_3)] + c_3``, which is the conditional risk in the
definition of the value functions. ``bayes_cascade(..., assume_independence=True)`` instead uses the
per-stage risk ``R_3(x; f_3*)``, the form these risks take under the conditional independence
``Y _||_ Z_{1:t-1} | (X, Z_t)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

NUM_CLASSES = 3
NUM_STAGES = 3
# Feature dimensions of (Z_1, Z_2, Z_3); Z_1 is empty in this task.
Z_DIMS = (0, 1, 1)


def gate(x: np.ndarray) -> np.ndarray:
    """``g(x) = sigmoid(10 (|x| - 0.15))``."""
    return 1.0 / (1.0 + np.exp(-10.0 * (np.abs(x) - 0.15)))


def logits(x: np.ndarray, z2: np.ndarray, z3: np.ndarray) -> np.ndarray:
    """Class logits ``l(x, z_2, z_3)``, broadcast over the inputs, shape ``(..., 3)``."""
    x, z2, z3 = np.broadcast_arrays(x, z2, z3)
    g = gate(x)
    return np.stack(
        [1.2 - 4.0 * np.abs(x), -0.2 + 2.2 * x + 3.0 * g * z2, -0.2 - 2.2 * x + 3.0 * g * z3],
        axis=-1,
    )


def theta(x: np.ndarray, z2: np.ndarray, z3: np.ndarray) -> np.ndarray:
    """Class probabilities ``softmax(l(x, z_2, z_3))``, shape ``(..., 3)``."""
    raw = logits(x, z2, z3)
    raw = raw - raw.max(axis=-1, keepdims=True)
    expd = np.exp(raw)
    return expd / expd.sum(axis=-1, keepdims=True)


def sample(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Draw ``n`` i.i.d. samples ``(x, z_2, z_3, y)``."""
    x, z2, z3 = rng.uniform(-1.0, 1.0, size=(3, n))
    cdf = theta(x, z2, z3).cumsum(axis=-1)
    y = (cdf < rng.random(n)[:, None]).sum(axis=-1)
    return {"x": x, "z2": z2, "z3": z3, "y": np.minimum(y, NUM_CLASSES - 1)}


@dataclass
class Posteriors:
    """Monte-Carlo posteriors on a grid of ``x`` values.

    Attributes:
        x: ``(G,)`` grid.
        p_x: ``(G, K)`` estimate of ``p(y | x)``.
        max_p_xz2: ``(G, M)`` values of ``max_y p(y | x, z_2)`` at ``M`` draws of ``Z_2``.
        max_p_xz3: ``(G, M)`` values of ``max_y p(y | x, z_3)`` at ``M`` draws of ``Z_3``.
        acc3_given_z2: ``(G, M)`` accuracy of ``f_3*`` given ``(x, z_2)`` at the ``Z_2``
            draws, ``E[theta_{f_3*(x, Z_3)}(x, z_2, Z_3)]``.
    """

    x: np.ndarray
    p_x: np.ndarray
    max_p_xz2: np.ndarray
    max_p_xz3: np.ndarray
    acc3_given_z2: np.ndarray


def monte_carlo_posteriors(x_grid: np.ndarray, n_mc: int, rng: np.random.Generator,
                           chunk: int = 500) -> Posteriors:
    """Estimate the posteriors with ``n_mc`` draws each of ``Z_2`` and ``Z_3``.

    For every ``x``, ``theta`` is evaluated on all ``n_mc x n_mc`` pairs
    ``(z_2, z_3)``. Averaging over ``z_3`` gives ``p(y | x, z_2)``, averaging over
    ``z_2`` gives ``p(y | x, z_3)``, and averaging over both gives ``p(y | x)``.
    A second pass evaluates the accuracy of ``f_3*`` given each draw of ``z_2``.
    """
    z2 = rng.uniform(-1.0, 1.0, n_mc)
    z3 = rng.uniform(-1.0, 1.0, n_mc)
    num_x = len(x_grid)
    p_x = np.zeros((num_x, NUM_CLASSES))
    max_p_xz2 = np.zeros((num_x, n_mc))
    max_p_xz3 = np.zeros((num_x, n_mc))
    acc3_given_z2 = np.zeros((num_x, n_mc))
    columns = np.arange(n_mc)
    for i, x in enumerate(x_grid):
        sum_over_z2 = np.zeros((n_mc, NUM_CLASSES))
        for start in range(0, n_mc, chunk):
            rows = theta(x, z2[start:start + chunk, None], z3[None, :])  # (chunk, n_mc, K)
            max_p_xz2[i, start:start + chunk] = rows.mean(axis=1).max(axis=-1)
            sum_over_z2 += rows.sum(axis=0)
        p_xz3 = sum_over_z2 / n_mc
        max_p_xz3[i] = p_xz3.max(axis=-1)
        p_x[i] = p_xz3.mean(axis=0)
        f3 = p_xz3.argmax(axis=-1)  # f_3*(x, z_3) at each draw of Z_3
        for start in range(0, n_mc, chunk):
            rows = theta(x, z2[start:start + chunk, None], z3[None, :])
            acc3_given_z2[i, start:start + chunk] = rows[:, columns, f3].mean(axis=1)
    return Posteriors(x=np.asarray(x_grid), p_x=p_x, max_p_xz2=max_p_xz2, max_p_xz3=max_p_xz3,
                      acc3_given_z2=acc3_given_z2)


def conditional_risks(post: Posteriors, costs: Sequence[float]) -> np.ndarray:
    """``R_t(x; f_t*) = 1 - E[max_y p(y | x, Z_t) | X = x] + c_t``, shape ``(G, 3)``."""
    expected_max = np.stack(
        [post.p_x.max(axis=-1), post.max_p_xz2.mean(axis=-1), post.max_p_xz3.mean(axis=-1)], axis=-1
    )
    return 1.0 - expected_max + np.asarray(costs)[None, :]


def bayes_flat(post: Posteriors, kappa: Sequence[float]) -> Dict[str, np.ndarray]:
    """Bayes-optimal flat router ``r*`` on the grid and the Bayes risk (``c = kappa``)."""
    risks = conditional_risks(post, kappa)
    return {"risks": risks, "router": risks.argmin(axis=-1), "bayes_risk": float(risks.min(axis=-1).mean())}


def bayes_cascade(post: Posteriors, kappa: Sequence[float],
                  assume_independence: bool = False) -> Dict[str, np.ndarray]:
    """Bayes-optimal first deferral ``d_1*`` on the grid and the Bayes risk.

    Uses the cumulative costs ``c = [k_1, k_1 + k_2, k_1 + k_2 + k_3]``. The rule
    defers when the stage-1 risk strictly exceeds the expected value of continuing,
    so ties exit. The stage-3 continuation risk is conditioned on the history ``(x, z_2)``
    by default. With ``assume_independence=True`` it is the per-stage risk of
    ``f_3*``, as under the conditional independence ``Y _||_ Z_{1:t-1} | (X, Z_t)``
    (see the module docstring).
    """
    costs = np.cumsum(kappa)
    risks = conditional_risks(post, costs)
    stage2 = 1.0 - post.max_p_xz2 + costs[1]            # R_2(x, z_2) at the Z_2 draws
    if assume_independence:
        stage3 = np.broadcast_to(risks[:, 2:3], stage2.shape)
    else:
        stage3 = 1.0 - post.acc3_given_z2 + costs[2]     # R_3(x, z_2), conditioned on the history
    value2 = np.minimum(stage2, stage3)                  # V_2(x, z_2)
    continue_value = value2.mean(axis=-1)                # E[V_2(x, Z_2) | X = x]
    defer = risks[:, 0] > continue_value
    value1 = np.minimum(risks[:, 0], continue_value)
    return {"risks": risks, "defer": defer.astype(np.int64), "continue_value": continue_value,
            "bayes_risk": float(value1.mean())}


def nearest_grid_index(x_grid: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Index of the closest point of a uniform grid for every entry of ``x``."""
    step = (x_grid[-1] - x_grid[0]) / (len(x_grid) - 1)
    return np.clip(np.rint((x - x_grid[0]) / step).astype(np.int64), 0, len(x_grid) - 1)
