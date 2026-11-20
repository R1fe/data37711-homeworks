"""Run the two white wine regression grids for homework 3, question 2(c)."""

import argparse
import csv
from itertools import product
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


SEED = 42
GRID_COLUMNS = [
    "model", "n_estimators", "max_depth", "min_samples_leaf", "max_features",
    "n_neighbors", "p", "weights", "validation_mse",
]


def forest_grid(x_train, y_train, x_val, y_val):
    for depth, leaf, features in product((6, 10, 16, None), (1, 4), (3, 5, 8, 11)):
        model = RandomForestRegressor(
            max_depth=depth, min_samples_leaf=leaf, max_features=features,
            warm_start=True, random_state=SEED, n_jobs=4,
        )
        # Reuse the first trees of each larger forest. The fixed seed keeps
        # each candidate identical to a forest fitted separately at that size.
        for trees in range(8, 129, 8):
            model.set_params(n_estimators=trees)
            model.fit(x_train, y_train)
            params = dict(n_estimators=trees, max_depth=depth,
                          min_samples_leaf=leaf, max_features=features)
            yield params, mean_squared_error(y_val, model.predict(x_val))


def knn_grid(x_train, y_train, x_val, y_val):
    for p, neighbors, weights in product((1, 2), range(1, 129), ("uniform", "distance")):
        params = dict(n_neighbors=neighbors, p=p, weights=weights)
        model = KNeighborsRegressor(**params).fit(x_train, y_train)
        yield params, mean_squared_error(y_val, model.predict(x_val))


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            # Distinguish RF's unlimited depth from fields unused by a model.
            writer.writerow({key: "None" if value is None else value for key, value in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wine_csv", type=Path, help="UCI Wine Quality, winequality-white.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    data = np.loadtxt(args.wine_csv, delimiter=";", skiprows=1)
    x, y = data[:, :-1], data[:, -1]
    x_pool, x_test, y_pool, y_test = train_test_split(x, y, test_size=0.20, random_state=SEED)
    x_train, x_val, y_train, y_val = train_test_split(
        x_pool, y_pool, test_size=0.125, random_state=SEED
    )
    print(f"scikit-learn {sklearn.__version__}, seed {SEED}, "
          f"train/validation/test {len(y_train)}/{len(y_val)}/{len(y_test)}", flush=True)

    scaler = StandardScaler().fit(x_train)
    knn_train = scaler.transform(x_train)
    knn_val = scaler.transform(x_val)
    experiments = [
        ("RF", forest_grid, RandomForestRegressor(random_state=SEED, n_jobs=4), x_train, x_val),
        ("KNN", knn_grid, KNeighborsRegressor(), knn_train, knn_val),
    ]
    grid_rows = []
    winners = []
    for name, grid, model, train_inputs, val_inputs in experiments:
        scores = list(grid(train_inputs, y_train, val_inputs, y_val))
        grid_rows.extend(dict(model=name, **params, validation_mse=mse) for params, mse in scores)
        best_params, best_mse = min(scores, key=lambda candidate: candidate[1])
        model.set_params(**best_params).fit(train_inputs, y_train)
        np.testing.assert_allclose(mean_squared_error(y_val, model.predict(val_inputs)), best_mse)
        winners.append((name, model, best_params, best_mse))

    selected_rows = []
    for name, model, params, val_mse in winners:
        test_inputs = scaler.transform(x_test) if name == "KNN" else x_test
        test_mse = mean_squared_error(y_test, model.predict(test_inputs))
        selected_rows.append(dict(model=name, **params, validation_mse=val_mse, test_mse=test_mse))
        print(f"{name}: {params}, validation MSE {val_mse:.8f}, test MSE {test_mse:.8f}")

    args.output_dir.mkdir(exist_ok=True, parents=True)
    write_csv(args.output_dir / "wine_grid.csv", GRID_COLUMNS, grid_rows)
    write_csv(args.output_dir / "wine_selected.csv", [*GRID_COLUMNS, "test_mse"], selected_rows)


if __name__ == "__main__":
    main()
