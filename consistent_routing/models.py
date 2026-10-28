"""Multi-model classifiers with a flat router or a cascade router.

Each classifier ``f_t`` sees the base feature ``x`` and its own additional
feature ``z_t`` (which may be empty, e.g. ``Z_1`` in the synthetic setup). It
outputs ``K - 1`` scores, completed to ``K`` sum-to-zero scores.

* :class:`FlatModel`: a router ``r(x)`` with a softmax output on the simplex
  chooses one classifier. Hard decision: ``argmax_t r_t(x)``.
* :class:`CascadeModel`: ``T - 1`` deferral modules ``d_t(x, z_{1:t})`` with a
  sigmoid output decide whether to stop at stage ``t`` or defer. Hard decision:
  exit at the first ``t`` with ``d_t < 0.5`` (the last stage always exits).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .losses import complete_scores

Features = Sequence[Optional[Tensor]]


def mlp(in_dim: int, out_dim: int, hidden: int = 128, n_hidden: int = 3,
        batch_norm: bool = True, out_batch_norm: bool = False) -> nn.Sequential:
    """``n_hidden`` blocks of Linear -> BatchNorm -> ReLU, then a linear output.

    ``out_batch_norm`` adds a BatchNorm on the output, as in the paper's router and
    deferral networks (``eq:router``, ``eq:cascade_router``).
    """
    layers: List[nn.Module] = []
    width = in_dim
    for _ in range(n_hidden):
        layers.append(nn.Linear(width, hidden))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden))
        layers.append(nn.ReLU())
        width = hidden
    layers.append(nn.Linear(width, out_dim))
    if out_batch_norm:
        layers.append(nn.BatchNorm1d(out_dim))
    return nn.Sequential(*layers)


def concat_features(x: Tensor, *zs: Optional[Tensor]) -> Tensor:
    """Concatenate ``x`` with the non-empty additional features."""
    return torch.cat([x] + [z for z in zs if z is not None], dim=-1)


def _router_kwargs(mlp_kwargs: dict, n_hidden: Optional[int], out_batch_norm: bool) -> dict:
    """``mlp_kwargs`` for a router or deferral network, which may be shallower than the classifiers."""
    kwargs = dict(mlp_kwargs, out_batch_norm=out_batch_norm)
    if n_hidden is not None:
        kwargs["n_hidden"] = n_hidden
    return kwargs


class ScoreClassifier(nn.Module):
    """Network with ``K - 1`` outputs, completed to ``K`` sum-to-zero scores."""

    def __init__(self, in_dim: int, num_classes: int, **mlp_kwargs):
        super().__init__()
        self.net = mlp(in_dim, num_classes - 1, **mlp_kwargs)

    def forward(self, inputs: Tensor) -> Tensor:
        return complete_scores(self.net(inputs))


class _MultiClassifier(nn.Module):
    """The ``T`` classifiers ``f_t(x, z_t)`` shared by both router types."""

    def __init__(self, x_dim: int, z_dims: Sequence[int], num_classes: int, **mlp_kwargs):
        super().__init__()
        self.z_dims = list(z_dims)
        self.num_classes = num_classes
        self.classifiers = nn.ModuleList(
            ScoreClassifier(x_dim + d, num_classes, **mlp_kwargs) for d in self.z_dims
        )

    @property
    def num_stages(self) -> int:
        return len(self.z_dims)

    def classifier_scores(self, x: Tensor, zs: Features) -> Tensor:
        """Scores of every classifier, shape ``(N, T, K)``."""
        return torch.stack(
            [f(concat_features(x, z)) for f, z in zip(self.classifiers, zs)], dim=1
        )


class FlatModel(_MultiClassifier):
    """Classifiers ``f_1, ..., f_T`` and a flat router ``r(x)`` on the simplex."""

    def __init__(self, x_dim: int, z_dims: Sequence[int], num_classes: int,
                 router_n_hidden: Optional[int] = None, router_out_batch_norm: bool = False, **mlp_kwargs):
        super().__init__(x_dim, z_dims, num_classes, **mlp_kwargs)
        self.router = mlp(x_dim, self.num_stages,
                          **_router_kwargs(mlp_kwargs, router_n_hidden, router_out_batch_norm))

    def forward(self, x: Tensor, zs: Features) -> Tuple[Tensor, Tensor]:
        """Returns scores ``(N, T, K)`` and routing probabilities ``(N, T)``."""
        return self.classifier_scores(x, zs), torch.softmax(self.router(x), dim=-1)

    @torch.no_grad()
    def decide(self, x: Tensor, zs: Features) -> Tuple[Tensor, Tensor]:
        """Hard decisions: chosen classifier and predicted class, each ``(N,)``."""
        scores, route_probs = self(x, zs)
        stage = route_probs.argmax(dim=1)
        pred = scores[torch.arange(len(stage)), stage].argmax(dim=-1)
        return stage, pred


def first_exit(defer_probs: Tensor) -> Tensor:
    """Index of the first stage with ``d_t < 0.5``; the last stage if none.

    Args:
        defer_probs: ``(N, T - 1)`` soft deferral probabilities.
    """
    num_samples = defer_probs.shape[0]
    stops = torch.cat(
        [defer_probs < 0.5, torch.ones(num_samples, 1, dtype=torch.bool, device=defer_probs.device)],
        dim=1,
    )
    # argmax returns the first maximal entry, i.e. the first stage that stops.
    return stops.to(torch.int64).argmax(dim=1)


class CascadeModel(_MultiClassifier):
    """Classifiers ``f_1, ..., f_T`` and deferral modules ``d_1, ..., d_{T-1}``.

    The deferral module at stage ``t`` sees the full history ``[x; z_1; ...; z_t]``
    and outputs the probability of deferring to stage ``t + 1``.
    """

    def __init__(self, x_dim: int, z_dims: Sequence[int], num_classes: int,
                 router_n_hidden: Optional[int] = None, router_out_batch_norm: bool = False, **mlp_kwargs):
        super().__init__(x_dim, z_dims, num_classes, **mlp_kwargs)
        history_dims = [x_dim + sum(self.z_dims[: t + 1]) for t in range(self.num_stages - 1)]
        deferral_kwargs = _router_kwargs(mlp_kwargs, router_n_hidden, router_out_batch_norm)
        self.deferrals = nn.ModuleList(mlp(d, 1, **deferral_kwargs) for d in history_dims)

    def forward(self, x: Tensor, zs: Features) -> Tuple[Tensor, Tensor]:
        """Returns scores ``(N, T, K)`` and deferral probabilities ``(N, T - 1)``."""
        scores = self.classifier_scores(x, zs)
        if not self.deferrals:
            return scores, x.new_zeros(x.shape[0], 0)
        defer = [
            torch.sigmoid(d(concat_features(x, *zs[: t + 1]))) for t, d in enumerate(self.deferrals)
        ]
        return scores, torch.cat(defer, dim=1)

    @torch.no_grad()
    def decide(self, x: Tensor, zs: Features) -> Tuple[Tensor, Tensor]:
        """Hard decisions: exit stage and predicted class, each ``(N,)``."""
        scores, defer_probs = self(x, zs)
        stage = first_exit(defer_probs)
        pred = scores[torch.arange(len(stage)), stage].argmax(dim=-1)
        return stage, pred
