"""Linear readouts and evaluation metrics."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

FloatArray = NDArray[np.float64]

DEFAULT_CS = (0.001, 0.01, 0.1, 1.0)
DEFAULT_ALPHAS = tuple(float(a) for a in np.logspace(-4, 4, 17))


def train_logistic_readout(
    X: FloatArray,
    y: NDArray[np.int64],
    Cs: tuple[float, ...] = DEFAULT_CS,
    cv: int = 5,
    seed: int = 0,
) -> Pipeline:
    """Standardise features and fit multinomial logistic regression.

    The inverse regularisation strength ``C`` is chosen by stratified k-fold
    cross-validation on the training set only.
    """
    min_class = int(np.bincount(y).min())
    folds = min(cv, min_class)
    best_c, best_score = Cs[0], -np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if folds >= 2 and len(Cs) > 1:
            splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
            for C in Cs:
                model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500))
                score = cross_val_score(model, X, y, cv=splitter).mean()
                if score > best_score:
                    best_c, best_score = C, score
        model = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, max_iter=2000))
        model.fit(X, y)
    return model


def train_ridge_readout(
    X: FloatArray, y: FloatArray, alphas: tuple[float, ...] = DEFAULT_ALPHAS
) -> Pipeline:
    """Standardise features and fit ridge regression; alpha picked by efficient LOO-CV."""
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    model.fit(X, y)
    return model


def accuracy(y_true: NDArray[np.int64], y_pred: NDArray[np.int64]) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def nrmse(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Root-mean-square error divided by the standard deviation of the target."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / (np.std(y_true) + 1e-12))
