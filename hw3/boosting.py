"""Five-fold learning curves for the two binary classification datasets in Q3."""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TREES = 1000
SEED = 42
CHECKPOINTS = (1, 10, 100, 300, 1000)
RESULTS = Path(__file__).resolve().parent / "results"
METRICS = (
    "n_train", "n_heldout", "train_error_count", "heldout_error_count",
    "train_error", "heldout_error", "train_log_loss", "heldout_log_loss",
)


def loss_sums(scores, labels):
    scores = scores.ravel()
    errors = np.count_nonzero((scores > 0) != labels)
    log_loss = np.logaddexp(0, (1 - 2 * labels) * scores).sum()
    return errors, log_loss


def compute_totals(train_scores, heldout_scores, y_train, y_heldout):
    train_errors, train_loss = loss_sums(train_scores, y_train)
    heldout_errors, heldout_loss = loss_sums(heldout_scores, y_heldout)
    return np.array((len(y_train), len(y_heldout), train_errors, heldout_errors,
                     train_loss, heldout_loss))


def metric_values(totals):
    train_n, heldout_n, train_errors, heldout_errors, train_loss, heldout_loss = totals
    return (int(train_n), int(heldout_n), int(train_errors), int(heldout_errors),
            train_errors / train_n, heldout_errors / heldout_n,
            train_loss / train_n, heldout_loss / heldout_n)


def evaluate(path):
    data = np.loadtxt(path, delimiter=",")
    X, y = data[:, 1:], data[:, 0].astype(int)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    boosting_totals = np.zeros((TREES, 6))
    logistic_totals = np.zeros(6)
    fold_rows, stage_rows, logistic_rows = [], [], []

    for fold, (train, heldout) in enumerate(splitter.split(X, y), start=1):
        X_train, X_heldout = X[train], X[heldout]
        y_train, y_heldout = y[train], y[heldout]
        model = GradientBoostingClassifier(
            n_estimators=TREES, learning_rate=0.1, max_depth=3,
            loss="log_loss", random_state=SEED,
        ).fit(X_train, y_train)
        stages = zip(model.staged_decision_function(X_train),
                     model.staged_decision_function(X_heldout))
        for stage, (train_scores, heldout_scores) in enumerate(stages, start=1):
            totals = compute_totals(train_scores, heldout_scores, y_train, y_heldout)
            boosting_totals[stage - 1] += totals
            fold_rows.append((path.name, fold, stage, *metric_values(totals)))

        linear = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        linear.fit(X_train, y_train)
        totals = compute_totals(linear.decision_function(X_train),
                                linear.decision_function(X_heldout), y_train, y_heldout)
        logistic_totals += totals
        logistic_rows.append((path.name, str(fold), *metric_values(totals)))

    print(f"\n{path.name}: {len(y)} rows, {y.sum()} positive labels")
    print("trees  train_error  heldout_error  heldout_log_loss")
    for stage, totals in enumerate(boosting_totals, start=1):
        values = metric_values(totals)
        stage_rows.append((path.name, stage, *values))
        if stage in CHECKPOINTS:
            print(f"{stage:5d}  {values[4]:11.4f}  {values[5]:13.4f}  {values[7]:16.4f}")
    baseline = metric_values(logistic_totals)
    logistic_rows.append((path.name, "pooled", *baseline))
    print(f"Logistic regression held-out error: {baseline[5]:.4f}")
    return fold_rows, stage_rows, logistic_rows


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs=2, type=Path, metavar="CSV")
    parser.add_argument("--output-dir", type=Path, default=RESULTS,
                        help="Result directory (default: results next to this script)")
    args = parser.parse_args()
    results = [evaluate(dataset) for dataset in args.datasets]
    folds, stages, baseline = [sum(parts, []) for parts in zip(*results)]
    args.output_dir.mkdir(exist_ok=True, parents=True)
    write_csv(args.output_dir / "boosting_folds.csv", ("dataset", "fold", "trees", *METRICS), folds)
    write_csv(args.output_dir / "boosting_stages.csv", ("dataset", "trees", *METRICS), stages)
    write_csv(args.output_dir / "logistic_baseline.csv", ("dataset", "fold", *METRICS), baseline)


if __name__ == "__main__":
    main()
