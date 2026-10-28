import pytest
import torch

from consistent_routing import (
    CascadeHingeLoss,
    FlatHingeLoss,
    complete_scores,
    cumulative_costs,
    exit_probabilities,
    stage_hinge,
)


def test_complete_scores_sum_to_zero():
    partial = torch.randn(5, 4)
    full = complete_scores(partial)
    assert full.shape == (5, 5)
    assert torch.allclose(full.sum(-1), torch.zeros(5), atol=1e-6)
    assert torch.equal(full[:, :4], partial)


@pytest.mark.parametrize(
    "scores, label, cost, expected",
    [
        # Score 1 on the true class and -1/(K-1) elsewhere is the hinge minimizer: loss 0.
        ([1.0, -0.5, -0.5], 0, 0.0, 0.0),
        # The cost enters as K/(K-1) * c = 1.5 * 0.1.
        ([1.0, -0.5, -0.5], 0, 0.1, 0.15),
        # Zero scores: each of the two wrong classes contributes 1/(K-1) = 0.5.
        ([0.0, 0.0, 0.0], 1, 0.0, 1.0),
        # K = 2: [s_2 + 1]_+ with s_2 = -0.3.
        ([0.3, -0.3], 0, 0.0, 0.7),
    ],
)
def test_stage_hinge_hand_values(scores, label, cost, expected):
    loss = stage_hinge(torch.tensor([scores]), torch.tensor([label]), cost)
    assert loss.item() == pytest.approx(expected)


def test_stage_hinge_batched_over_classifiers():
    scores = complete_scores(torch.randn(6, 3, 4))
    y = torch.randint(0, 5, (6,))
    costs = torch.tensor([0.0, 0.1, 0.2])
    batched = stage_hinge(scores, y, costs)
    looped = torch.stack([stage_hinge(scores[:, t], y, costs[t]) for t in range(3)], dim=1)
    assert torch.allclose(batched, looped)


def test_cumulative_costs():
    assert torch.allclose(cumulative_costs([0.0, 0.05, 0.05]), torch.tensor([0.0, 0.05, 0.1]))


def test_flat_loss_is_router_weighted_sum():
    torch.manual_seed(0)
    scores = complete_scores(torch.randn(8, 3, 2))
    y = torch.randint(0, 3, (8,))
    route = torch.softmax(torch.randn(8, 3), dim=-1)
    kappa = [0.0, 0.05, 0.06]
    manual = sum(route[:, t] * stage_hinge(scores[:, t], y, kappa[t]) for t in range(3)).mean()
    assert FlatHingeLoss(kappa)(scores, route, y).item() == pytest.approx(manual.item(), rel=1e-6)


def test_flat_loss_with_one_hot_router_is_the_chosen_hinge():
    scores = complete_scores(torch.randn(4, 3, 2))
    y = torch.tensor([0, 1, 2, 0])
    route = torch.nn.functional.one_hot(torch.tensor([0, 2, 1, 2]), 3).float()
    kappa = [0.0, 0.05, 0.06]
    chosen = [stage_hinge(scores[i:i + 1, t], y[i:i + 1], kappa[t]) for i, t in enumerate([0, 2, 1, 2])]
    expected = torch.cat(chosen).mean()
    assert FlatHingeLoss(kappa)(scores, route, y).item() == pytest.approx(expected.item(), rel=1e-6)


def test_exit_probabilities():
    defer = torch.rand(10, 2)
    probs = exit_probabilities(defer)
    assert torch.allclose(probs.sum(-1), torch.ones(10), atol=1e-6)
    d1, d2 = defer[:, 0], defer[:, 1]
    assert torch.allclose(probs, torch.stack([1 - d1, d1 * (1 - d2), d1 * d2], dim=1), atol=1e-6)
    # Never deferring exits at stage 1; always deferring reaches the last stage.
    assert torch.equal(exit_probabilities(torch.zeros(1, 2)), torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.equal(exit_probabilities(torch.ones(1, 2)), torch.tensor([[0.0, 0.0, 1.0]]))


def test_cascade_loss_with_hard_deferrals_uses_cumulative_cost_of_exit_stage():
    scores = complete_scores(torch.randn(3, 3, 2))
    y = torch.tensor([0, 1, 2])
    defer = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])  # exits at stages 1, 2, 3
    kappa = [0.0, 0.05, 0.05]
    cum = [0.0, 0.05, 0.1]
    expected = torch.cat([stage_hinge(scores[i:i + 1, i], y[i:i + 1], cum[i]) for i in range(3)]).mean()
    assert CascadeHingeLoss(kappa)(scores, defer, y).item() == pytest.approx(expected.item(), rel=1e-6)


def test_losses_are_differentiable():
    partial = torch.randn(5, 3, 2, requires_grad=True)
    logits = torch.randn(5, 3, requires_grad=True)
    defer_logits = torch.randn(5, 2, requires_grad=True)
    y = torch.randint(0, 3, (5,))
    scores = complete_scores(partial)
    loss = FlatHingeLoss([0.0, 0.1, 0.2])(scores, logits.softmax(-1), y)
    loss = loss + CascadeHingeLoss([0.0, 0.1, 0.1])(scores, defer_logits.sigmoid(), y)
    loss.backward()
    for tensor in (partial, logits, defer_logits):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
