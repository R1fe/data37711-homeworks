# DATA 37711 coursework

## Homework 2 - Problem 1(c,e)

`hw2/p1.py` reproduces the taxi-duration experiment: two Gaussian MLP candidates, validation-based model selection, constant/linear Gaussian baselines, and the training-prevalence probability baseline. It evaluates duration RMSE, Gaussian NLL, 90% prediction-interval coverage, Brier score, binary log-loss, and accuracy at threshold 0.5.

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
