"""Homework 2, Problem 2: target encoding, embeddings, and their combination.

Run from the repository root:
    python hw2/p2.py --data /path/to/retail.csv
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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
import torch
from torch import nn

SEED = 42
DATA_SHA256 = "5da6724e4ee820f19e1bdb5076fd24355a10f8e75159062824c85f17ea052597"
torch.set_num_threads(4)


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_indices(n):
    train, rest = train_test_split(np.arange(n), test_size=.30, random_state=SEED)
    val, test = train_test_split(rest, test_size=.50, random_state=SEED)
    return train, val, test


def tensor(x, integer=False):
    return torch.as_tensor(np.asarray(x), dtype=torch.long if integer else torch.float32)


def fit_network(model, train_inputs, target, val_inputs, val_target, loss_fn,
                epochs=250, patience=25, batch_size=1024, lr=1e-3, weight_decay=1e-4):
    """Restore the checkpoint with the lowest validation loss."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_epoch, best_state = float("inf"), 0, None
    history = []
    n = len(target)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for idx in perm.split(batch_size):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(*(x[idx] for x in train_inputs)), target[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            opt.step()
        model.eval()
        with torch.no_grad():
            tr = float(loss_fn(model(*train_inputs), target))
            va = float(loss_fn(model(*val_inputs), val_target))
        history.append({"epoch": epoch, "train": tr, "validation": va})
        if epoch == 1 or epoch % 10 == 0:
            print(f"  epoch={epoch} train={tr:.6f} val={va:.6f} elapsed={time.perf_counter()-started:.1f}s", flush=True)
        if va < best - 1e-7:
            best, best_epoch = va, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    print(f"  selected epoch={best_epoch}, validation loss={best:.6f}", flush=True)
    return history, best_epoch


class Dense(nn.Module):
    def __init__(self, inputs, widths, outputs=1, activation=nn.ReLU):
        super().__init__()
        layers = []
        for width in widths:
            layers.extend([nn.Linear(inputs, width), activation()])
            inputs = width
        layers.append(nn.Linear(inputs, outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def fit_target_maps(frame, y):
    # Arithmetic means of the original Quantity, computed only within training rows.
    return {c:pd.DataFrame({"key":frame[c].to_numpy(),"target":y}).groupby("key")["target"].mean()
            for c in frame}, float(np.mean(y))


def apply_target_maps(frame, maps, default):
    return np.column_stack([frame[c].map(maps[c]).fillna(default).to_numpy(float) for c in frame])


class RetailNet(nn.Module):
    def __init__(self, cardinalities, mode):
        super().__init__()
        self.mode = mode
        if mode != "target_only":
            self.embeddings = nn.ModuleList([nn.Embedding(n+1,16,padding_idx=0) for n in cardinalities])
        dim = 2 if mode == "target_only" else (32 if mode == "embeddings" else 34)
        self.head = Dense(dim, [64,32])

    def forward(self, ids, te):
        if self.mode == "target_only":
            x = te
        else:
            x = torch.cat([emb(ids[:,j]) for j,emb in enumerate(self.embeddings)],dim=1)
            if self.mode == "combined":
                x = torch.cat([x,te],dim=1)
        return self.head(x)


def experiment(data_path, out):
    df = pd.read_csv(data_path, dtype={"StockCode":"string","CustomerID":"string"})
    if len(df) != 397924 or set(df.columns) != {"StockCode", "CustomerID", "Quantity"}:
        raise ValueError("Expected the corrected course retail.csv (397,924 rows, three columns)")
    features = ["StockCode","CustomerID"]
    frame = df[features].fillna("__MISSING__")
    y = df.Quantity.to_numpy(float)
    if not np.isfinite(y).all() or np.any(y <= 0):
        raise ValueError("Quantity must be finite and positive; inspect source data")
    it,iv,ie = split_indices(len(y))
    frames = [frame.iloc[i].reset_index(drop=True) for i in [it,iv,ie]]
    oof = np.zeros((len(it),2))
    for fit, held in KFold(n_splits=5,shuffle=True,random_state=SEED).split(it):
        maps,default = fit_target_maps(frames[0].iloc[fit],y[it][fit])
        oof[held] = apply_target_maps(frames[0].iloc[held],maps,default)
    maps,default = fit_target_maps(frames[0],y[it])
    te = [oof]+[apply_target_maps(f,maps,default) for f in frames[1:]]
    te = [np.log1p(a) for a in te]
    tc,ts = te[0].mean(0),np.maximum(te[0].std(0),1e-8)
    te = [tensor((a-tc)/ts) for a in te]
    vocab = {c:{v:j+1 for j,v in enumerate(sorted(frames[0][c].unique()))} for c in features}
    ids = [tensor(np.column_stack([f[c].map(vocab[c]).fillna(0).to_numpy(int) for c in features]),True)
           for f in frames]
    z = np.log1p(y)
    zc,zs = z[it].mean(),z[it].std()
    yy = [tensor(((z[i]-zc)/zs)[:,None]) for i in [it,iv,ie]]
    scores,selection,histories = {},{},{}
    for mode in ["target_only","embeddings","combined"]:
        print(f"Retail mode={mode}", flush=True)
        seed_all()
        net = RetailNet([len(vocab[c]) for c in features],mode)
        hist,epoch = fit_network(net,[ids[0],te[0]],yy[0],[ids[1],te[1]],yy[1],
                                nn.MSELoss(),epochs=120,patience=15,batch_size=2048)
        histories[mode] = pd.DataFrame(hist)
        with torch.no_grad():
            zp = net(ids[2],te[2]).squeeze(1).numpy()*zs+zc
            zv = net(ids[1],te[1]).squeeze(1).numpy()*zs+zc
        pred = np.maximum(np.expm1(np.clip(zp,0,30)),0.)
        scores[mode] = {"rmsle":float(np.sqrt(mean_squared_error(z[ie],np.log1p(pred)))),
                        "mae":float(mean_absolute_error(y[ie],pred)),
                        "rmse":float(np.sqrt(mean_squared_error(y[ie],pred)))}
        selection[mode] = {"best_epoch":epoch,"validation_rmsle":float(np.sqrt(mean_squared_error(z[iv],np.maximum(zv,0))))}
    baseline = np.full(len(ie),np.expm1(zc))
    scores["constant_log_mean"] = {"rmsle":float(np.sqrt(mean_squared_error(z[ie],np.log1p(baseline)))),
                                    "mae":float(mean_absolute_error(y[ie],baseline)),
                                    "rmse":float(np.sqrt(mean_squared_error(y[ie],baseline)))}
    plot_curves(histories, zs, out/"learning_curves.png")
    return {"rows":len(df),"splits":list(map(len,[it,iv,ie])),
            "full_cardinality":{c:int(frame[c].nunique()) for c in features},
            "train_cardinality":{c:len(vocab[c]) for c in features},
            "test_unknown_fraction":{c:float((ids[2][:,j]==0).float().mean()) for j,c in enumerate(features)},
            "quantity_quantiles":{str(q):float(np.quantile(y,q)) for q in [.0,.5,.9,.99,.999,1.]},
            "selection":selection,"test_metrics":scores}


def plot_curves(histories, target_scale, path):
    fig, ax = plt.subplots(figsize=(9, 3.2), constrained_layout=True)
    for mode, label, color in [("target_only", "Target encoding", "#62758a"),
                               ("embeddings", "Embeddings", "#167c80"),
                               ("combined", "Combined", "#c16b33")]:
        h = histories[mode]
        ax.plot(h.epoch, np.sqrt(h.validation)*target_scale,
                label=f"{label}: validation", color=color)
        ax.plot(h.epoch, np.sqrt(h.train)*target_scale, "--", alpha=.6, color=color)
    ax.set(xlabel="Epoch", ylabel="Root mean squared log-target error",
           title="Retail learning curves (dashed: training)")
    ax.legend(fontsize=8, ncol=3)
    fig.savefig(path, dpi=210)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Corrected course retail.csv")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent/"results"/"p2")
    args = parser.parse_args()
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError("Dataset SHA-256 differs from the corrected course retail.csv")
    args.output.mkdir(parents=True, exist_ok=True)
    results = experiment(args.data, args.output)
    (args.output/"metrics.json").write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame.from_dict(results["test_metrics"], orient="index").rename_axis(
        "model").to_csv(args.output/"metrics.csv")
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
