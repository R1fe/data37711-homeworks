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
