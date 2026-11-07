"""Homework 2, Problem 3: six separate one-factor hyperparameter sweeps.

Run from the repository root:
    python hw2/p3.py --data-dir /path/to/course/data
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import torch
from torch import nn

SEED = 42
RKS_BASE = {"h": 256, "sigma": 1.0, "alpha": 0.1}
RKS_GRIDS = {"h": [32, 64, 128, 256, 512, 1024],
             "sigma": [0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
             "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]}
MLP_BASE = {"width": 64, "lr": 0.003, "batch_size": 256}
MLP_GRIDS = {"width": [16, 32, 64, 128, 256],
             "lr": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03],
             "batch_size": [32, 64, 128, 256, 600]}
SHA256 = {
    "train.csv": "becc881f01b6c34450b95036c115f92682fed9fb5e1e48b81bb681f8cf672e8e",
    "val.csv": "2d96164fa2019e8333e93f7a9ba96bfb606ec60b2fa4c485d0b45501753fbac6",
    "test.csv": "3b0f5fe8ae346f5dff4f805de7893ce190f331a2c8871e0f857cf66dcab00857",
}
torch.set_num_threads(4)


def save_json(value, path):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def tensor(value):
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


def feature_bank():
    """Nested features across h; h=256 reproduces the original random draw."""
    rng = np.random.default_rng(SEED)
    w = rng.normal(size=(2, 256))
    b = rng.uniform(0, 2*np.pi, size=256)
    extra = max(RKS_GRIDS["h"]) - 256
    return (np.concatenate([w, rng.normal(size=(2, extra))], axis=1),
            np.concatenate([b, rng.uniform(0, 2*np.pi, size=extra)]))


class RKS:
    def __init__(self, config, bank):
        self.w = bank[0][:, :config["h"]] / config["sigma"]
        self.b = bank[1][:config["h"]]
        self.regression = Ridge(alpha=config["alpha"])

    def features(self, x):
        phase = x @ self.w + self.b
        return np.concatenate([np.sin(phase), np.cos(phase)], axis=1)

    def fit(self, x, y):
        self.regression.fit(self.features(x), y)
        return self

    def predict(self, x):
        return self.regression.predict(self.features(x))


def fit_mlp(config, x, y, xv, yv, history_path):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model = nn.Sequential(nn.Linear(2, config["width"]), nn.Tanh(),
                          nn.Linear(config["width"], 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    best, best_epoch, checkpoint = float("inf"), 0, None
    history = []
    started = time.perf_counter()
    for epoch in range(1, 1501):
        model.train()
        for indices in torch.randperm(len(y)).split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x[indices]), y[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss for {config}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            training_loss = float(loss_fn(model(x), y))
            validation_loss = float(loss_fn(model(xv), yv))
        history.append({"epoch": epoch, "train_standardized_mse": training_loss,
                        "validation_standardized_mse": validation_loss})
        if validation_loss < best - 1e-7:
            best, best_epoch = validation_loss, epoch
            checkpoint = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 250 == 0:
            print(f"  epoch={epoch}, val={validation_loss:.7f}, "
                  f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
        if epoch - best_epoch >= 100:
            break
    model.load_state_dict(checkpoint)
    model.eval()
    pd.DataFrame(history).to_csv(history_path, index=False)
    return model, best_epoch, epoch


def predict_mlp(model, x, center, scale):
    with torch.no_grad():
        return model(tensor(x/3)).squeeze(1).numpy()*scale + center


def metrics(y, yp, yv, yvp):
    return {"train_mse": float(mean_squared_error(y, yp)),
            "validation_mse": float(mean_squared_error(yv, yvp))}


def plot_sweeps(rows, family, grids, out):
    labels = {"h": "Feature pairs h (D = 2h)", "sigma": "Kernel scale sigma",
              "alpha": "Ridge penalty lambda", "width": "Hidden width H",
              "lr": "Learning rate", "batch_size": "Batch size"}
    summary, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    for r, parameter in enumerate(grids):
        records = [row for row in rows if row["parameter"] == parameter]
        values = [row["value"] for row in records]
        best = min(records, key=lambda row: row["validation_mse"])
        for c, metric in enumerate(["train_mse", "validation_mse"]):
            individual, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
            for target in [ax, axes[r, c]]:
                target.plot(values, [row[metric] for row in records], "o-", color="#167c80")
                target.scatter([best["value"]], [best[metric]], s=70,
                               facecolors="none", edgecolors="#c16b33", linewidths=1.8)
                target.set(xlabel=labels[parameter], ylabel="MSE",
                           title=f"{family.upper()}: {metric.replace('_', ' ')}")
                target.set_xscale("log")
                target.grid(alpha=.2)
            individual.savefig(out/f"{family}_{parameter}_{metric}.png", dpi=180)
            plt.close(individual)
    summary.savefig(out/f"{family}_sweeps.png", dpi=180)
    plt.close(summary)


def run(data_dir, out):
    frames = []
    for filename, digest in SHA256.items():
        path = data_dir/filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Unexpected course data: {filename}")
        frames.append(pd.read_csv(path))
    train, validation, test = frames
    assert len(train) == 600 and len(validation) == len(test) == 200
    assert "y" not in test and {"x1", "x2", "y"} <= set(train) & set(validation)
    x, xv, xe = [frame[["x1", "x2"]].to_numpy(float) for frame in frames]
    y, yv = train.y.to_numpy(float), validation.y.to_numpy(float)
    assert all(np.isfinite(a).all() for a in [x, xv, xe, y, yv])
    center, scale = y.mean(), y.std()
    tx, tv = tensor(x/3), tensor(xv/3)
    ty, tyv = tensor(((y-center)/scale)[:, None]), tensor(((yv-center)/scale)[:, None])
    bank = feature_bank()
    sweeps, winners, caches = {}, {}, {}
    for family, base, grids in [("rks", RKS_BASE, RKS_GRIDS), ("mlp", MLP_BASE, MLP_GRIDS)]:
        rows, cache, group_winners = [], {}, {}
        for parameter, values in grids.items():
            group = []
            for value in values:
                config = {**base, parameter: value}
                key = tuple(sorted(config.items()))
                if key not in cache:
                    print(f"{family.upper()} {config}", flush=True)
                    extra = {}
                    if family == "rks":
                        model = RKS(config, bank).fit(x, y)
                        score = metrics(y, model.predict(x), yv, model.predict(xv))
                    else:
                        model, best_epoch, stopped_epoch = fit_mlp(
                            config, tx, ty, tv, tyv,
                            out/f"mlp_{parameter}_{value}_history.csv")
                        score = metrics(y, predict_mlp(model, x, center, scale),
                                        yv, predict_mlp(model, xv, center, scale))
                        extra = {"best_epoch": best_epoch, "stopped_epoch": stopped_epoch}
                    cache[key] = {"model": model, "config": config, **score, **extra}
                    print(f"  completed: {score}", flush=True)
                result = cache[key]
                row = {"parameter": parameter, "value": value, **config,
                       **{k: v for k, v in result.items() if k not in {"model", "config"}}}
                rows.append(row)
                group.append(row)
                pd.DataFrame(rows).to_csv(out/f"{family}_sweeps.csv", index=False)
            group_winners[parameter] = min(group, key=lambda row: row["validation_mse"])
        sweeps[family], winners[family], caches[family] = rows, group_winners, cache
        plot_sweeps(rows, family, grids, out)

    selected = {family: min(cache.values(), key=lambda r: r["validation_mse"])
                for family, cache in caches.items()}
    # Refit the best actually evaluated configuration, retaining the validation split.
    rks = RKS(selected["rks"]["config"], bank).fit(x, y)
    mlp, epoch, stopped = fit_mlp(selected["mlp"]["config"], tx, ty, tv, tyv,
                                 out/"mlp_refit_history.csv")
    refit = {"rks": metrics(y, rks.predict(x), yv, rks.predict(xv)),
             "mlp": {**metrics(y, predict_mlp(mlp, x, center, scale),
                               yv, predict_mlp(mlp, xv, center, scale)),
                     "best_epoch": epoch, "stopped_epoch": stopped}}
    for family in refit:
        np.testing.assert_allclose(refit[family]["validation_mse"],
                                   selected[family]["validation_mse"], rtol=1e-7, atol=1e-7)
    for family, predictions in [("rks", rks.predict(xe)),
                                ("mlp", predict_mlp(mlp, xe, center, scale))]:
        assert len(predictions) == 200 and np.isfinite(predictions).all()
        np.savetxt(out/f"{family}_test_predictions.csv", predictions, delimiter=",")

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    points = ax.scatter(x[:, 0], x[:, 1], c=y, s=10, cmap="viridis")
    fig.colorbar(points, ax=ax, label="y")
    ax.set(xlabel="x1", ylabel="x2", title="Training data", xlim=(-3, 3), ylim=(-3, 3))
    fig.savefig(out/"toy_training_data.png", dpi=180)
    plt.close(fig)
    gx, gy = np.meshgrid(np.linspace(-3, 3, 121), np.linspace(-3, 3, 121))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    surfaces = [rks.predict(grid).reshape(gx.shape),
                predict_mlp(mlp, grid, center, scale).reshape(gx.shape)]
    low = min(y.min(), *(surface.min() for surface in surfaces))
    high = max(y.max(), *(surface.max() for surface in surfaces))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, family, surface in zip(axes, ["RKS", "One-hidden-layer MLP"], surfaces):
        im = ax.pcolormesh(gx, gy, surface, shading="auto", cmap="viridis", vmin=low, vmax=high)
        ax.set(xlabel="x1", ylabel="x2", title=family, aspect="equal")
    fig.colorbar(im, ax=axes, label="Predicted y")
    fig.savefig(out/"toy_fitted_surfaces.png", dpi=180)
    plt.close(fig)
    results = {"seed": SEED, "rows": {"train": 600, "validation": 200, "test": 200},
               "rks_base": RKS_BASE, "mlp_base": MLP_BASE,
               "rks_grids": RKS_GRIDS, "mlp_grids": MLP_GRIDS,
               "unique_fits": {family: len(cache) for family, cache in caches.items()},
               "sweep_winners": winners,
               "selected": {family: {k: v for k, v in r.items() if k != "model"}
                            for family, r in selected.items()},
               "refit": refit}
    save_json(results, out/"metrics.json")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent/"results"/"p3")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run(args.data_dir, args.output)
