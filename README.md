# Consistent routing: hinge losses for jointly training classifiers and routers

This package implements the cost-sensitive hinge surrogate losses for jointly training `T` classifiers with a
single-stage (flat) router or with a cascade router. It also includes the synthetic experiment of the paper.


## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Requires Python 3.9 or later and PyTorch 2.0 or later. The code runs on CPU.

## Tests

```bash
pytest
```

## Synthetic experiment

```bash
python experiments/synthetic_routing.py            # paper settings, a few minutes on a laptop CPU
python experiments/synthetic_routing.py --quick    # small run for a smoke test
```

The script trains a flat router and a cascade router with `kappa = [0, 0.05, 0.05]`. The flat cost vector is
`c = [0, 0.05, 0.05]`, and the cascade uses the cumulative `c = [0, 0.05, 0.1]`. It writes to `--out`
(default `results/synthetic`):

* `results.json`: test 0-1-c risk (error rate, average cost), Monte-Carlo Bayes risk, the fraction of test points
  where the learned decision matches the Bayes-optimal one, the training history and the configuration.
* `flat_routing.pdf` and `cascade_routing.pdf`: the two panels of Fig. 2. The top shows the conditional risks
  `R_t(x; f_t*)` with the Bayes-optimal decision shaded. The bottom shows the test points colored by the learned
  routing probabilities.

## Training on precomputed LLM embeddings

`experiments/embedding_routing.py` trains a flat router or a cascade on prompt and answer embeddings that you
provide. It does not compute embeddings. Each sample needs a prompt embedding `x`, the embedding `z_t` of each
model's answer to that prompt, and the label. By default the files are read from `<emb-dir>/<dataset>/`:

```
embeddings/<dataset>/prompt.npy        (N, d)   prompt embeddings
embeddings/<dataset>/labels.npy        (N,)     labels in {0, ..., K-1}
embeddings/<dataset>/<model>.npy       (N, d)   answer embeddings of each model
```

The presets set the number of classes (`GSM8K`: 4, `CommonsenseQA`: 5) and the models of each family, in
cascade order (`llama`: llama-3.2-1b, llama-3.3-70b, llama-3.1-405b; `qwen`: qwen2.5-1.5b, qwen2.5-32b,
qwen2.5-72b). Any file can be overridden, and `--models` replaces the family preset.

```bash
python experiments/embedding_routing.py --dataset GSM8K --family llama --router flat --kappa 0 0.05 0.07
python experiments/embedding_routing.py --dataset CommonsenseQA --family qwen --router cascade \
    --answer-emb qwen2.5-72b=/path/to/qwen72b.npy --kappa 0 0.05 0.06
```

## Using the losses with your own features

Each sample is a base feature `x`, one additional feature `z_t` per classifier (`None` when it is empty), and a
label `y`. `fit` trains either model with mini-batches and early stopping on the validation surrogate loss.

```python
import torch
from consistent_routing import CascadeHingeLoss, CascadeModel, FlatHingeLoss, FlatModel, zero_one_c_risk
from consistent_routing.train import TrainConfig, fit

kappa = [0.0, 0.05, 0.06]                     # cost of each classifier, in cascade order
x_dim, z_dims, num_classes = 768, [768, 768, 768], 5

def split(n):                                  # replace with your own features and labels
    return (torch.randn(n, x_dim), [torch.randn(n, d) for d in z_dims], torch.randint(0, num_classes, (n,)))

train, val, test = split(800), split(100), split(100)

arch = dict(hidden=128, n_hidden=2, router_n_hidden=1, router_out_batch_norm=True)
adam = dict(lr=1e-3, optimizer="adam", weight_decay=0.0, batch_size=32, patience=20)

model = FlatModel(x_dim, z_dims, num_classes, **arch)
fit(model, FlatHingeLoss(kappa), train, val, TrainConfig(max_epochs=100, **adam))
stage, pred = model.decide(test[0], test[1])
print(zero_one_c_risk(pred, test[2], stage, kappa))

cascade = CascadeModel(x_dim, z_dims, num_classes, **arch)
fit(cascade, CascadeHingeLoss(kappa), train, val, TrainConfig(max_epochs=1000, **adam))
stage, pred = cascade.decide(test[0], test[1])
print(zero_one_c_risk(pred, test[2], stage, torch.cumsum(torch.tensor(kappa), 0)))
```

