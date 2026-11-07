# Homework 2

## Problem 1(c,e)

[`p1.py`](p1.py) reproduces the taxi-duration experiment: two Gaussian MLP candidates, validation-based model selection, constant/linear Gaussian baselines, and the training-prevalence probability baseline. It evaluates duration RMSE, Gaussian NLL, 90% prediction-interval coverage, Brier score, binary log-loss, and accuracy at threshold 0.5.

### Run

Use **Python 3.11** on CPU. From the cloned repository root:

```bash
python -m venv .venv
# Activate: source .venv/bin/activate (Linux), or .venv\Scripts\activate (Windows).
python -m pip install -r requirements.txt
python hw2/p1.py --data /path/to/taxi_trips.parquet
```

The required data are the corrected `taxi_trips.parquet` supplied with Homework 2 on Canvas (October 28). Obtain this file from the course; course data are not redistributed here. The script verifies its SHA-256:

```text
6a713446584efe41bb161f77f068331b964a04c84abd3b69164bc2bdb585e3dc
```

An agent needs only this repository, Python 3.11, the installed requirements, and the supplied dataset. No notebook state, local paths, credentials, or saved model weights are required. Defaults write `metrics.json`, `metrics.csv`, and `diagnostics.png` to `hw2/results/p1/`; override with `--output PATH`.

### Report results

Seed 42; split sizes 366,487 / 78,533 / 78,534. Target normalization is fitted on training rows only; supplied feature scaling is preserved. Candidates have hidden widths 64/64 and 128/64. Both use ReLU, Gaussian NLL, Adam (learning rate 0.001, weight decay 0.0001), batches of 2,048, maximum 150 epochs and early-stopping patience 20. The selected model is 128/64 at epoch 84.

| Model | RMSE (minutes) | Gaussian NLL | 90% coverage | Brier | Binary log-loss | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Constant Gaussian | 7.757 | 3.4675 | 94.27% | 0.004457 | 0.055276 | 99.554% |
| Linear Gaussian | 7.544 | 3.4398 | 93.76% | 0.004455 | 0.045934 | 99.554% |
| Training prevalence | - | - | - | 0.004437 | 0.028572 | 99.554% |
| MLP | 3.923 | 2.7551 | 93.85% | 0.002792 | 0.021746 | 99.675% |

The diagnostic figure contains long-trip probability calibration and predicted mean versus observed duration. No test labels are used for model selection. Small floating-point differences across platforms or library versions are possible; the pinned environment was evaluated with Windows / Python 3.11.5 / four CPU threads.

## Problem 2 - Categorical representations

[`p2.py`](p2.py) reproduces the reported retail experiments: target encoding, learned embeddings, their combination, and a constant log-mean baseline. It prints feature cardinalities and reports held-out RMSLE, MAE, and RMSE.

Use the same Python 3.11 CPU environment and root `requirements.txt` as above. From the repository root:

```bash
python hw2/p2.py --data /path/to/retail.csv
```

Obtain the corrected `retail.csv` from Homework 2 on Canvas. Course data are not redistributed here. The script verifies SHA-256 `5da6724e4ee820f19e1bdb5076fd24355a10f8e75159062824c85f17ea052597` and writes `metrics.json`, `metrics.csv`, and `learning_curves.png` to `hw2/results/p2/`; override with `--output PATH`.

Seed 42; train/validation/test sizes 278,546 / 59,689 / 59,689. Both identifiers are categorical strings. Training target encodings use five-fold out-of-fold raw-quantity means; validation/test maps use training labels only, with the training global mean for unseen categories. Encoded means are log-transformed and standardized. Each embedding has 16 dimensions; unseen categories use zero vectors. Model input sizes are 2, 32, and 34.

All three networks have a 64/32 ReLU head and minimize squared error on the standardized `log1p(Quantity)` target. Training-only statistics define all standardization. Adam uses learning rate 0.001, weight decay 0.0001, batch size 2,048, gradient clipping at 10, maximum 120 epochs, and early-stopping patience 15. The best validation checkpoint is restored; no test labels are used for model selection. Predictions are transformed back to nonnegative quantities.

| Model | Best epoch | Validation RMSLE | Test RMSLE | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|---:|
| Constant log-mean | - | - | 0.99832 | 10.206 | 41.363 |
| Target encoding | 51 | 0.62229 | 0.62303 | 6.899 | 32.424 |
| Embeddings | 43 | 0.52484 | 0.52837 | 5.909 | 29.380 |
| Combined | 27 | 0.51762 | 0.52123 | 5.798 | 30.111 |

The combined model has the lowest test RMSLE and MAE; embeddings alone have the lowest raw-scale RMSE. The learning curves plot log-target error by epoch (solid: validation; dashed: training). Platform-dependent floating-point variation is possible, as for Problem 1.

## Problem 3 - Random features and learned features

[`p3.py`](p3.py) reproduces the expanded report: separate one-factor sweeps of all six suggested hyperparameters, validation-based selection, refitting, fitted surfaces, and predictions for the unlabeled test set. Use the same Python 3.11 CPU environment and root `requirements.txt` as above. From the repository root:

```bash
python hw2/p3.py --data-dir /path/to/course/data
```

The data directory must contain the corrected course `train.csv`, `val.csv`, and `test.csv`. The script verifies the SHA-256 checksums recorded in the code and preserves the supplied 600/200/200 splits. Course data are not redistributed here. Outputs default to `hw2/results/p3/`; override with `--output PATH`.

### Independent sweeps

Each sweep changes only its named parameter around a fixed reference configuration. RKS uses `(h, sigma, lambda) = (256, 1, 0.1)`; MLP uses `(hidden width, learning rate, batch size) = (64, 0.003, 256)`.

| Model | Scanned parameter | Values |
|---|---|---|
| RKS | Feature pairs h (D = 2h) | 32, 64, 128, 256, 512, 1024 |
| RKS | Kernel scale sigma | 0.2, 0.4, 0.7, 1, 1.5, 2, 3, 5, 10 |
| RKS | Ridge penalty lambda | 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100 |
| MLP | Hidden width | 16, 32, 64, 128, 256 |
| MLP | Learning rate | 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03 |
| MLP | Batch size | 32, 64, 128, 256, 600 |

Seed 42 and four CPU threads are used throughout. RKS fits unnormalized sine/cosine features with a sum-squared-error ridge objective and an unpenalized intercept. Random draws are shared across scale and penalty sweeps; feature-count comparisons use nested subsets of one feature bank.

The MLP has exactly one Tanh hidden layer. Inputs are divided by 3; targets use training-only standardization. Adam uses weight decay 0.00001, gradient clipping at 10, maximum 1500 epochs, and validation patience 100. The best validation checkpoint is restored. Reported MSE is on the original target scale. Batch-size comparisons use the same epoch limit, so their optimizer-update counts differ.

Shared reference configurations are reused across sweeps: 23 RKS and 16 MLP sweep entries correspond to 21 and 14 distinct fits. The best complete configuration actually evaluated for each model is then refitted, keeping validation separate. This is not a joint grid search; independent sweep winners are not combined without evaluation.

| Sweep | Best value | Training MSE | Validation MSE |
|---|---:|---:|---:|
| RKS h | 64 | 0.031215 | 0.037196 |
| RKS sigma | 3 | 0.032666 | 0.036309 |
| RKS lambda | 10 | 0.031101 | 0.038224 |
| MLP width | 32 | 0.034449 | 0.035932 |
| MLP learning rate | 0.01 | 0.034267 | 0.035843 |
| MLP batch size | 64 | 0.033819 | 0.035609 |

The final RKS configuration is `h=256, sigma=3, lambda=0.1` (validation MSE 0.036308719). The final MLP configuration is `width=64, learning_rate=0.003, batch_size=64`, selected at epoch 882 (validation MSE 0.035609375).

### Generated files

The script writes complete sweep tables (`rks_sweeps.csv`, `mlp_sweeps.csv`), selection/refit metrics (`metrics.json`), twelve individual training/validation plots, combined sweep figures (`rks_sweeps.png`, `mlp_sweeps.png`), training-data and fitted-surface plots, and per-epoch MLP histories. It also writes `rks_test_predictions.csv` and `mlp_test_predictions.csv`, each with exactly 200 finite predictions, no header or index, in the original test row order.

The test set has no labels, so no test MSE is reported. Validation is used for model selection and is not an independent final test. Small platform-dependent floating-point differences are possible. No notebook state or saved model weights are required.
