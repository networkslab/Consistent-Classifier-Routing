"""Mini-batch training with early stopping on the validation surrogate loss."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

# (x, [z_1, ..., z_T], y); an entry z_t may be None when Z_t is empty.
Split = Tuple[Tensor, Sequence[Optional[Tensor]], Tensor]


@dataclass
class TrainConfig:
    lr: float = 1e-3
    optimizer: str = "adamw"  # "adamw" or "adam"
    weight_decay: float = 1e-2  # AdamW default; use 0 with "adam"
    batch_size: int = 512
    max_epochs: int = 50
    patience: int = 2
    seed: int = 0


def _take(split: Split, idx: Tensor) -> Split:
    x, zs, y = split
    return x[idx], [None if z is None else z[idx] for z in zs], y[idx]


def surrogate_loss(model: nn.Module, loss_fn: nn.Module, split: Split) -> float:
    """Surrogate loss on a whole split, with batch normalization in eval mode."""
    model.eval()
    with torch.no_grad():
        x, zs, y = split
        scores, router_out = model(x, zs)
        return loss_fn(scores, router_out, y).item()


def fit(model: nn.Module, loss_fn: nn.Module, train: Split, val: Split,
        config: TrainConfig) -> Dict[str, List[float]]:
    """Train ``model`` on ``loss_fn`` with Adam or AdamW and early stopping.

    ``loss_fn(scores, router_out, y)`` is a :class:`~consistent_routing.losses.FlatHingeLoss`
    or :class:`~consistent_routing.losses.CascadeHingeLoss`. Training stops once the
    validation surrogate loss has not improved for ``config.patience`` epochs, and the
    weights of the best epoch are restored.
    """
    generator = torch.Generator().manual_seed(config.seed)
    optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    if config.optimizer not in optimizers:
        raise ValueError(f"unknown optimizer {config.optimizer!r}; use 'adam' or 'adamw'")
    optimizer = optimizers[config.optimizer](model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    num_train = train[2].shape[0]
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val, best_state, stale = float("inf"), copy.deepcopy(model.state_dict()), 0

    for _ in range(config.max_epochs):
        model.train()
        order = torch.randperm(num_train, generator=generator)
        running, seen = 0.0, 0
        for start in range(0, num_train, config.batch_size):
            idx = order[start:start + config.batch_size]
            if len(idx) < 2:  # batch normalization needs at least two samples
                continue
            x, zs, y = _take(train, idx)
            scores, router_out = model(x, zs)
            loss = loss_fn(scores, router_out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * len(idx)
            seen += len(idx)

        val_loss = surrogate_loss(model, loss_fn, val)
        history["train_loss"].append(running / max(seen, 1))
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return history
