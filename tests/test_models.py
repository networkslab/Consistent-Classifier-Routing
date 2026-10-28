import torch

from consistent_routing import CascadeModel, FlatModel, first_exit


def _inputs(n=16, x_dim=4, z_dims=(0, 3, 2)):
    x = torch.randn(n, x_dim)
    zs = [None if d == 0 else torch.randn(n, d) for d in z_dims]
    return x, zs


def test_flat_model_outputs():
    torch.manual_seed(0)
    x, zs = _inputs()
    model = FlatModel(4, (0, 3, 2), num_classes=5, hidden=16, n_hidden=2)
    scores, route = model(x, zs)
    assert scores.shape == (16, 3, 5)
    assert torch.allclose(scores.sum(-1), torch.zeros(16, 3), atol=1e-5)
    assert route.shape == (16, 3)
    assert torch.allclose(route.sum(-1), torch.ones(16), atol=1e-6)
    assert (route >= 0).all()

    model.eval()
    stage, pred = model.decide(x, zs)
    scores, route = model(x, zs)
    assert torch.equal(stage, route.argmax(-1))
    assert torch.equal(pred, scores[torch.arange(16), stage].argmax(-1))


def test_cascade_model_outputs():
    torch.manual_seed(0)
    x, zs = _inputs()
    model = CascadeModel(4, (0, 3, 2), num_classes=5, hidden=16, n_hidden=2)
    # The deferral module at stage t sees [x; z_1; ...; z_t].
    assert model.deferrals[0][0].in_features == 4
    assert model.deferrals[1][0].in_features == 4 + 3
    scores, defer = model(x, zs)
    assert scores.shape == (16, 3, 5)
    assert defer.shape == (16, 2)
    assert ((defer > 0) & (defer < 1)).all()

    model.eval()
    stage, pred = model.decide(x, zs)
    scores, defer = model(x, zs)
    assert torch.equal(stage, first_exit(defer))
    assert torch.equal(pred, scores[torch.arange(16), stage].argmax(-1))


def test_first_exit_rule():
    defer = torch.tensor([[0.4, 0.9], [0.6, 0.3], [0.6, 0.7], [0.5, 0.2]])
    # Exit at the first stage with d < 0.5; d = 0.5 defers; the last stage always exits.
    assert first_exit(defer).tolist() == [0, 1, 2, 1]


def test_single_stage_cascade_has_no_deferral():
    x, zs = _inputs(z_dims=(2,))
    model = CascadeModel(4, (2,), num_classes=3, hidden=8, n_hidden=1)
    scores, defer = model(x, zs)
    assert defer.shape == (16, 0)
    model.eval()
    stage, _ = model.decide(x, zs)
    assert (stage == 0).all()
