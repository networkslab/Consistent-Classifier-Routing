import json
import sys
from pathlib import Path

import numpy as np
import pytest

from consistent_routing import synthetic

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import synthetic_routing  # noqa: E402


def test_gate_and_logits():
    assert synthetic.gate(np.array(0.15)) == pytest.approx(0.5)
    x, z2, z3 = 0.4, -0.3, 0.7
    g = synthetic.gate(np.array(x))
    expected = [1.2 - 4 * abs(x), -0.2 + 2.2 * x + 3 * g * z2, -0.2 - 2.2 * x + 3 * g * z3]
    assert np.allclose(synthetic.logits(np.array(x), np.array(z2), np.array(z3)), expected)


def test_theta_is_a_distribution():
    rng = np.random.default_rng(0)
    probs = synthetic.theta(*rng.uniform(-1, 1, size=(3, 100)))
    assert probs.shape == (100, 3)
    assert np.allclose(probs.sum(-1), 1.0)
    assert (probs > 0).all()


def test_sample_labels_follow_theta():
    rng = np.random.default_rng(0)
    data = synthetic.sample(200_000, rng)
    assert set(np.unique(data["y"])) <= {0, 1, 2}
    # Empirical class frequencies match the average of theta over the samples.
    expected = synthetic.theta(data["x"], data["z2"], data["z3"]).mean(axis=0)
    observed = np.bincount(data["y"], minlength=3) / len(data["y"])
    assert np.allclose(observed, expected, atol=5e-3)


def test_posteriors_and_bayes_quantities():
    grid = np.linspace(-1, 1, 21)
    post = synthetic.monte_carlo_posteriors(grid, n_mc=200, rng=np.random.default_rng(0))
    assert np.allclose(post.p_x.sum(-1), 1.0)
    # Extra information cannot increase the conditional risk at zero cost.
    risks = synthetic.conditional_risks(post, [0.0, 0.0, 0.0])
    assert (risks[:, 1] <= risks[:, 0] + 1e-9).all()
    assert (risks[:, 2] <= risks[:, 0] + 1e-9).all()

    kappa = [0.0, 0.05, 0.05]
    flat = synthetic.bayes_flat(post, kappa)
    cascade = synthetic.bayes_cascade(post, kappa)
    always_f1 = float(synthetic.conditional_risks(post, kappa)[:, 0].mean())
    assert flat["bayes_risk"] <= always_f1 + 1e-12
    assert cascade["bayes_risk"] <= always_f1 + 1e-12
    assert set(np.unique(flat["router"])) <= {0, 1, 2}
    assert set(np.unique(cascade["defer"])) <= {0, 1}


def test_stage3_accuracy_averages_to_the_marginal_formula():
    # Tower property: averaging the accuracy of f_3* given (x, z_2) over Z_2 gives
    # E[max_y p(y | x, Z_3)], exactly for the same Monte-Carlo draws.
    post = synthetic.monte_carlo_posteriors(np.array([-0.3, 0.2, 0.7]), n_mc=150, rng=np.random.default_rng(1))
    assert np.allclose(post.acc3_given_z2.mean(axis=-1), post.max_p_xz3.mean(axis=-1))


def test_history_based_and_per_stage_continuation_values_differ_at_x_07():
    # The two options of bayes_cascade give different stage-1 decisions at x = 0.7.
    post = synthetic.monte_carlo_posteriors(np.array([0.7]), n_mc=600, rng=np.random.default_rng(2))
    kappa = [0.0, 0.05, 0.05]
    assert synthetic.bayes_cascade(post, kappa, assume_independence=True)["defer"][0] == 1
    assert synthetic.bayes_cascade(post, kappa)["defer"][0] == 0


def test_nearest_grid_index():
    grid = np.linspace(-1, 1, 5)  # step 0.5
    idx = synthetic.nearest_grid_index(grid, np.array([-1.0, -0.74, -0.76, 0.1, 1.0, 1.3]))
    assert idx.tolist() == [0, 1, 0, 2, 4, 4]


def test_experiment_smoke(tmp_path):
    args = synthetic_routing.parse_args([
        "--n-train", "600", "--n-val", "200", "--n-test", "300", "--epochs", "1",
        "--batch-size", "128", "--hidden", "16", "--n-mc", "40", "--grid", "21", "--out", str(tmp_path),
    ])
    results = synthetic_routing.run(args)
    for name in ("flat", "cascade"):
        assert 0.0 <= results[name]["agreement_with_bayes"] <= 1.0
        assert results[name]["test"]["risk"] == pytest.approx(
            results[name]["test"]["error_rate"] + results[name]["test"]["avg_cost"])
        assert (tmp_path / f"{name}_routing.pdf").exists()
    assert results["cascade"]["costs"] == pytest.approx([0.0, 0.05, 0.1])
    assert json.loads((tmp_path / "results.json").read_text())["flat"]["costs"] == pytest.approx([0.0, 0.05, 0.05])
