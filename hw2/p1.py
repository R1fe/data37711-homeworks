"""Homework 2, Problem 1(c,e): Gaussian regression and P(duration < 45).

Run from the repository root:
    python hw2/p1.py --data /path/to/taxi_trips.parquet
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import torch
from torch import nn

SEED = 42
FEATURES = ["pickup_longitude", "pickup_latitude", "dropoff_longitude",
            "dropoff_latitude", "min_of_day", "day_of_week", "day_of_year"]
DATA_SHA256 = "6a713446584efe41bb161f77f068331b964a04c84abd3b69164bc2bdb585e3dc"


class GaussianMLP(nn.Module):
    def __init__(self, widths):
        super().__init__()
        layers, inputs = [], 7
        for width in widths:
            layers.extend([nn.Linear(inputs, width), nn.ReLU()])
            inputs = width
        layers.append(nn.Linear(inputs, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def nll(output, y):
    mu = output[:, :1]
    sigma = nn.functional.softplus(output[:, 1:2]) + 1e-4
    return (torch.log(sigma) + .5 * ((y - mu) / sigma)**2
            + .5 * math.log(2 * math.pi)).mean()


def fit(widths, x, y, xv, yv):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model = GaussianMLP(widths)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_epoch, checkpoint = float("inf"), 0, None
    for epoch in range(1, 151):
        model.train()
        for indices in torch.randperm(len(y)).split(2048):
            optimizer.zero_grad(set_to_none=True)
            loss = nll(model(x[indices]), y[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = float(nll(model(xv), yv))
        if validation < best - 1e-7:
            best, best_epoch = validation, epoch
            checkpoint = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 10 == 0:
            print(f"widths={widths}, epoch={epoch}, val_nll={validation:.6f}", flush=True)
        if epoch - best_epoch >= 20:
            break
    model.load_state_dict(checkpoint)
    return model, best_epoch, best


def probability_scores(y, probability):
    event = (y < 45.).astype(float)
    # Float64 clipping avoids log(0) from float32 rounding near one.
    probability = np.asarray(probability, dtype=np.float64)
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    return {
        "brier": float(np.mean((probability - event)**2)),
        "binary_log_loss": float(-np.mean(event * np.log(clipped)
                                         + (1 - event) * np.log1p(-clipped))),
        "accuracy_at_half": float(np.mean((probability >= .5) == event)),
    }


def gaussian_scores(y, mu, sigma):
    mu, sigma = np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)
    probability = ndtr((45 - mu) / sigma)
    return {
        "rmse_minutes": float(np.sqrt(np.mean((y - mu)**2))),
        "gaussian_nll": float(np.mean(np.log(sigma) + .5 * ((y - mu) / sigma)**2
                                      + .5 * np.log(2 * np.pi))),
        "central_90pct_coverage": float(np.mean(np.abs(y - mu) <= 1.6448536269514722 * sigma)),
        **probability_scores(y, probability),
    }


def plot_diagnostics(y, mu, sigma, output):
    long_probability = 1 - ndtr((45 - mu.astype(float)) / sigma.astype(float))
    long_event = y >= 45
    bins = [0, .001, .005, .01, .02, .05, .1, .2, .4, .6, .8, 1.0000001]
    predicted, observed = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (long_probability >= lo) & (long_probability < hi)
        if mask.any():
            predicted.append(long_probability[mask].mean())
            observed.append(long_event[mask].mean())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axes[0].plot(predicted, observed, "o-", label="Held-out test bins")
    axes[0].set(xlabel="Predicted P(duration >= 45)", ylabel="Observed fraction >= 45",
                title="Long-trip probability calibration",
                xlim=(0, max(.25, max(predicted) * 1.1)),
                ylim=(0, max(.25, max(observed) * 1.1)))
    axes[0].legend(fontsize=8)
    selected = np.random.default_rng(SEED).choice(len(y), min(5000, len(y)), replace=False)
    axes[1].scatter(y[selected], mu[selected], s=3, alpha=.2)
    axes[1].plot([0, 80], [0, 80], "--", color="gray")
    axes[1].set(xlabel="Observed minutes", ylabel="Predicted mean minutes",
                title="Conditional mean (5,000 test trips)", xlim=(0, 80), ylim=(0, 80))
    fig.savefig(output / "diagnostics.png", dpi=210)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Corrected course taxi_trips.parquet")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "p1")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error("Supply the course taxi_trips.parquet with --data; it is not included in this repository.")
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != DATA_SHA256:
        parser.error("Dataset checksum differs from the corrected course file used in the report.")
    torch.set_num_threads(4)
    df = pd.read_parquet(args.data)
    x, y = df[FEATURES].to_numpy(float), df["trip_duration"].to_numpy(float)
    if len(df) != 523554 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Unexpected taxi dataset shape or non-finite observations")
    train, rest = train_test_split(np.arange(len(y)), test_size=.30, random_state=SEED)
    validation, test = train_test_split(rest, test_size=.50, random_state=SEED)
    center, scale = y[train].mean(), y[train].std()
    xt, xv, xe = [torch.tensor(x[i], dtype=torch.float32) for i in [train, validation, test]]
    yt, yv = [torch.tensor(((y[i] - center) / scale)[:, None], dtype=torch.float32)
              for i in [train, validation]]
    selection, models = [], []
    for widths in [(64, 64), (128, 64)]:
        model, epoch, loss = fit(widths, xt, yt, xv, yv)
        selection.append({"widths": list(widths), "best_epoch": epoch,
                          "validation_nll": float(loss + np.log(scale))})
        models.append(model)
    best = int(np.argmin([r["validation_nll"] for r in selection]))
    with torch.no_grad():
        raw = models[best](xe)
        mu = raw[:, 0].numpy() * scale + center
        sigma = (nn.functional.softplus(raw[:, 1]) + 1e-4).numpy() * scale

    linear = Ridge(alpha=1.).fit(x[train], y[train])
    residual_sigma = np.sqrt(np.mean((y[train] - linear.predict(x[train]))**2))
    scores = {
        "constant_gaussian": gaussian_scores(y[test], center, scale),
        "linear_gaussian": gaussian_scores(y[test], linear.predict(x[test]), residual_sigma),
        "heteroscedastic_mlp": gaussian_scores(y[test], mu, sigma),
        "training_prevalence": probability_scores(y[test], np.mean(y[train] < 45.)),
    }
    result = {"seed": SEED, "data_sha256": DATA_SHA256,
              "splits": dict(zip(["train", "validation", "test"], map(len, [train, validation, test]))),
              "selection": selection, "selected": selection[best], "test_metrics": scores}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame.from_dict(scores, orient="index").rename_axis("model").to_csv(args.output / "metrics.csv")
    plot_diagnostics(y[test], mu, sigma, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
