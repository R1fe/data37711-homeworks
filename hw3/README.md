# DATA 37711 · Homework 3

[Written solutions and derivations](solution.tex)

## 2(c) · White wine quality

Each model was trained on 3,428 wines and tuned on 490. The table gives the lowest validation mean squared error (MSE) from 512 configurations per model and the corresponding MSE on 980 held-out wines. The split and model seed was 42.

| Model | Selected configuration | Validation MSE | Test MSE |
| --- | --- | ---: | ---: |
| Random forest | 120 trees, unlimited depth, minimum leaf size 1, 3 candidate features per split | 0.29898 | 0.34925 |
| K-nearest neighbors | 25 neighbors, Manhattan distance, inverse-distance weights | 0.29874 | 0.36855 |

The nearest-neighbor model had a 0.00024 advantage on validation MSE. The forest had a 0.01930 advantage on test MSE. [All 1,024 validation scores](results/wine_grid.csv) · [Selected models and test scores](results/wine_selected.csv) · [Grid search code](wine_search.py)

## 3 · Gradient boosting

Each dataset has 500 observations. We predicted the binary first column from the remaining five using stratified five-fold cross-validation with shuffling and seed 42. Each boosting model used depth-3 trees and a learning rate of 0.1. The reported cross-validation (CV) errors pool the prediction for each observation from the fold that held it out.

| Trees | dset0 train error | dset0 CV error | dset0 CV log loss | dset1 train error | dset1 CV error | dset1 CV log loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 42.60% | 42.6% | 0.662 | 29.05% | 32.0% | 0.648 |
| 10 | 15.90% | 26.0% | 0.571 | 12.75% | 20.4% | 0.478 |
| 100 | 0.80% | 17.2% | 0.401 | 1.35% | 19.8% | 0.409 |
| 300 | 0.00% | 16.0% | 0.440 | 0.00% | 20.0% | 0.513 |
| 1,000 | 0.00% | 16.8% | 0.781 | 0.00% | 20.8% | 0.957 |

Among these five checkpoints, 300 trees gave the lowest CV classification error on dset0 and 100 trees gave the lowest on dset1. The CSVs also report every intermediate stage from 1 to 1,000. Logistic regression on the same folds had error rates of 35.8% and 17.4%, respectively.

[All stages by fold](results/boosting_folds.csv) · [All stages pooled across folds](results/boosting_stages.csv) · [Logistic regression baseline](results/logistic_baseline.csv) · [Boosting code](boosting.py)

## Reproduce

Use Python 3.11. Run these commands from this folder with the paths to the three input files. The dataset for question 2(c) is the semicolon-delimited `winequality-white.csv` from [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality). The other two CSVs are the headerless files supplied for question 3.

```sh
python -m pip install -r requirements.txt
python wine_search.py /path/to/winequality-white.csv
python boosting.py /path/to/dset0-1.csv /path/to/dset1-1.csv
```

Both scripts write their tables to `results/` by default. Pass `--output-dir` to write a separate run elsewhere.
