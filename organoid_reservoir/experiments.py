"""Task runners shared by the CLI, the sweep command and the notebook.

Each runner returns a list of flat result rows
``{task, model, seed, n_neurons, metric, value, seconds}`` so everything ends up
in one tidy CSV.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import data
from .config import ExperimentConfig
from .encoding import ElectrodeEncoder, pad_sequences
from .esn import EchoStateNetwork
from .features import mean_std_features, pool_sequence_states, resample_sequences
from .lstm import lstm_classify, lstm_forecast, torch_available
from .readout import accuracy, nrmse, train_logistic_readout, train_ridge_readout
from .reservoir import SpikingReservoir

FloatArray = NDArray[np.float64]
Row = dict[str, Any]

VOWEL_MODELS = (
    "spiking",
    "spiking_stdp",
    "spiking_dc",
    "esn",
    "lstm",
    "logreg_mean_std",
    "logreg_resampled",
)
TS_MODELS = ("spiking", "spiking_stdp", "spiking_dc", "esn", "lstm", "linear_ar")
TS_TASKS = ("henon", "narma10", "mackey_glass")
NOISE_OFFSET = 10_000


def _row(task: str, model: str, seed: int, n: int, metric: str, value: float, t0: float) -> Row:
    return {
        "task": task,
        "model": model,
        "seed": seed,
        "n_neurons": n,
        "metric": metric,
        "value": float(value),
        "seconds": round(time.perf_counter() - t0, 3),
    }


def build_encoder(
    cfg: ExperimentConfig,
    n_inputs: int,
    fit_frames: FloatArray,
    electrodes_per_input: int | None = None,
    mode: str | None = None,
) -> ElectrodeEncoder:
    """Create and fit an electrode encoder, optionally overriding a few fields."""
    ecfg = copy.deepcopy(cfg.encoder)
    if electrodes_per_input is not None:
        ecfg.electrodes_per_input = electrodes_per_input
    if mode is not None:
        ecfg.mode = mode
    return ElectrodeEncoder(n_inputs, ecfg).fit(fit_frames)


def build_spiking(
    cfg: ExperimentConfig, encoder: ElectrodeEncoder, seed: int, n_neurons: int | None = None
) -> SpikingReservoir:
    rcfg = copy.deepcopy(cfg.reservoir)
    if n_neurons is not None:
        rcfg.n_neurons = n_neurons
    return SpikingReservoir(rcfg, encoder, seed=seed)


def build_esn(
    cfg: ExperimentConfig, encoder: ElectrodeEncoder, seed: int, n_neurons: int | None = None
) -> EchoStateNetwork:
    ecfg = copy.deepcopy(cfg.esn)
    if n_neurons is not None:
        ecfg.n_neurons = n_neurons
    return EchoStateNetwork(ecfg, encoder, seed=seed)


def stdp_weight_change(res: SpikingReservoir) -> float:
    """Relative Frobenius change of the E->E weights since construction."""
    ne = res.n_exc
    w0 = res.W_init[:ne, :ne]
    return float(np.linalg.norm(res.W[:ne, :ne] - w0) / (np.linalg.norm(w0) + 1e-12))


# ---------------------------------------------------------------------------
# Japanese Vowels
# ---------------------------------------------------------------------------


def vowels_seed(
    cfg: ExperimentConfig,
    ds: data.SequenceDataset,
    seed: int,
    models: Iterable[str] = VOWEL_MODELS,
    n_neurons: int | None = None,
) -> list[Row]:
    """Run the requested models on the vowels split for one seed."""
    models = list(models)
    rows: list[Row] = []
    Xtr, ltr = pad_sequences(ds.X_train)
    Xte, lte = pad_sequences(ds.X_test)
    frames = np.concatenate(ds.X_train)
    n_res = n_neurons or cfg.reservoir.n_neurons
    n_esn = n_neurons or cfg.esn.n_neurons

    def score_states(Str: NDArray[np.float32], Ste: NDArray[np.float32]) -> float:
        Ftr = pool_sequence_states(Str, ltr)
        Fte = pool_sequence_states(Ste, lte)
        clf = train_logistic_readout(Ftr, ds.y_train, seed=seed)
        return accuracy(ds.y_test, clf.predict(Fte))

    for mode, name in (("pulse", "spiking"), ("current", "spiking_dc")):
        if name not in models and not (name == "spiking" and "spiking_stdp" in models):
            continue
        enc = build_encoder(cfg, ds.n_features, frames, mode=mode)
        res = build_spiking(cfg, enc, seed, n_neurons)
        if name in models:
            t0 = time.perf_counter()
            res.reseed_noise(seed + NOISE_OFFSET)
            acc = score_states(res.transform(Xtr, ltr), res.transform(Xte, lte))
            rows.append(_row("vowels", name, seed, n_res, "accuracy", acc, t0))
        if name == "spiking" and "spiking_stdp" in models:
            t0 = time.perf_counter()
            for _ in range(cfg.stdp_epochs):
                order = np.random.default_rng(seed).permutation(len(Xtr))
                res.transform(Xtr[order], ltr[order], plastic=True, batch_size=cfg.stdp_batch_size)
            res.reseed_noise(seed + NOISE_OFFSET)
            acc = score_states(res.transform(Xtr, ltr), res.transform(Xte, lte))
            rows.append(_row("vowels", "spiking_stdp", seed, n_res, "accuracy", acc, t0))
            rows.append(
                _row(
                    "vowels",
                    "spiking_stdp",
                    seed,
                    n_res,
                    "weight_change",
                    stdp_weight_change(res),
                    t0,
                )
            )

    if "esn" in models:
        t0 = time.perf_counter()
        enc = build_encoder(cfg, ds.n_features, frames)
        esn = build_esn(cfg, enc, seed, n_neurons)
        acc = score_states(esn.transform(Xtr, ltr), esn.transform(Xte, lte))
        rows.append(_row("vowels", "esn", seed, n_esn, "accuracy", acc, t0))

    if "lstm" in models and cfg.lstm.enabled and torch_available():
        t0 = time.perf_counter()
        pred = lstm_classify(ds.X_train, ds.y_train, ds.X_test, cfg.lstm, seed=seed)
        rows.append(
            _row("vowels", "lstm", seed, cfg.lstm.hidden, "accuracy", accuracy(ds.y_test, pred), t0)
        )

    if "logreg_mean_std" in models:
        t0 = time.perf_counter()
        clf = train_logistic_readout(mean_std_features(ds.X_train), ds.y_train, seed=seed)
        acc = accuracy(ds.y_test, clf.predict(mean_std_features(ds.X_test)))
        rows.append(_row("vowels", "logreg_mean_std", seed, 0, "accuracy", acc, t0))

    if "logreg_resampled" in models:
        t0 = time.perf_counter()
        clf = train_logistic_readout(resample_sequences(ds.X_train, 8), ds.y_train, seed=seed)
        acc = accuracy(ds.y_test, clf.predict(resample_sequences(ds.X_test, 8)))
        rows.append(_row("vowels", "logreg_resampled", seed, 0, "accuracy", acc, t0))
    return rows


# ---------------------------------------------------------------------------
# Time-series tasks
# ---------------------------------------------------------------------------


@dataclass
class SeriesTask:
    """Input/target pair with train/test boundaries."""

    name: str
    u: FloatArray
    y: FloatArray
    washout: int
    n_train: int

    @property
    def split(self) -> int:
        return self.washout + self.n_train


def make_series_task(cfg: ExperimentConfig, name: str, seed: int = 0) -> SeriesTask:
    """Build a one-input time-series task. The series itself does not depend on the seed
    for Henon and Mackey-Glass (deterministic systems); NARMA input is seeded."""
    ts = cfg.timeseries
    total = ts.washout + ts.n_train + ts.n_test
    if name == "henon":
        x = data.henon_map(total + 1)
        u, y = x[:-1], x[1:]
    elif name == "narma10":
        u, y = data.narma10(total, seed=seed)
    elif name == "mackey_glass":
        h = ts.mackey_horizon
        x = data.mackey_glass(total + h)
        u, y = x[:-h], x[h:]
    elif name == "memory":
        u = data.memory_capacity_input(total, seed=seed)
        y = u.copy()
    else:
        raise ValueError(f"unknown time-series task {name!r}")
    return SeriesTask(name, u, y, ts.washout, ts.n_train)


def series_states(model: SpikingReservoir | EchoStateNetwork, u: FloatArray) -> NDArray[np.float32]:
    """Run a reservoir on a single 1-D input series; return (T, n) states."""
    return model.transform(u[None, :, None])[0]


def _readout_features(states: NDArray[np.float32], u: FloatArray) -> FloatArray:
    return np.concatenate([np.asarray(states, dtype=np.float64), u[:, None]], 1)


def fit_series_readout(task: SeriesTask, states: NDArray[np.float32]) -> tuple[float, FloatArray]:
    """Ridge readout on [state, input]; returns test NRMSE and test predictions."""
    X = _readout_features(states, task.u)
    s = task.split
    model = train_ridge_readout(X[task.washout : s], task.y[task.washout : s])
    pred = model.predict(X[s:])
    return nrmse(task.y[s:], pred), pred


def lagged(u: FloatArray, lags: int) -> FloatArray:
    """Matrix whose columns are u[t], u[t-1], ..., u[t-lags+1] (zero padded)."""
    out = np.zeros((len(u), lags))
    for k in range(lags):
        out[k:, k] = u[: len(u) - k]
    return out


def timeseries_seed(
    cfg: ExperimentConfig,
    task_name: str,
    seed: int,
    models: Iterable[str] = TS_MODELS,
    n_neurons: int | None = None,
    keep_predictions: bool = False,
) -> tuple[list[Row], dict[str, FloatArray]]:
    """Run the requested models on one time-series task for one seed."""
    models = list(models)
    task = make_series_task(cfg, task_name, seed)
    s = task.split
    fit_frames = task.u[:s, None]
    k = cfg.timeseries.electrodes_per_input
    rows: list[Row] = []
    preds: dict[str, FloatArray] = {"target": task.y[s:]} if keep_predictions else {}
    n_res = n_neurons or cfg.reservoir.n_neurons

    if "spiking" in models or "spiking_stdp" in models:
        enc = build_encoder(cfg, 1, fit_frames, electrodes_per_input=k)
        res = build_spiking(cfg, enc, seed, n_neurons)
        if "spiking" in models:
            t0 = time.perf_counter()
            res.reseed_noise(seed + NOISE_OFFSET)
            err, p = fit_series_readout(task, series_states(res, task.u))
            rows.append(_row(task_name, "spiking", seed, n_res, "nrmse", err, t0))
            if keep_predictions:
                preds["spiking"] = p
        if "spiking_stdp" in models:
            t0 = time.perf_counter()
            for _ in range(cfg.stdp_epochs):
                res.transform(task.u[None, :s, None], plastic=True)
            res.reseed_noise(seed + NOISE_OFFSET)
            err, p = fit_series_readout(task, series_states(res, task.u))
            rows.append(_row(task_name, "spiking_stdp", seed, n_res, "nrmse", err, t0))
            rows.append(
                _row(
                    task_name,
                    "spiking_stdp",
                    seed,
                    n_res,
                    "weight_change",
                    stdp_weight_change(res),
                    t0,
                )
            )
            if keep_predictions:
                preds["spiking_stdp"] = p

    if "spiking_dc" in models:
        t0 = time.perf_counter()
        enc = build_encoder(cfg, 1, fit_frames, electrodes_per_input=k, mode="current")
        res = build_spiking(cfg, enc, seed, n_neurons)
        res.reseed_noise(seed + NOISE_OFFSET)
        err, p = fit_series_readout(task, series_states(res, task.u))
        rows.append(_row(task_name, "spiking_dc", seed, n_res, "nrmse", err, t0))
        if keep_predictions:
            preds["spiking_dc"] = p

    if "esn" in models:
        t0 = time.perf_counter()
        enc = build_encoder(cfg, 1, fit_frames)
        esn = build_esn(cfg, enc, seed, n_neurons)
        err, p = fit_series_readout(task, series_states(esn, task.u))
        rows.append(_row(task_name, "esn", seed, n_neurons or cfg.esn.n_neurons, "nrmse", err, t0))
        if keep_predictions:
            preds["esn"] = p

    if "lstm" in models and cfg.lstm.enabled and torch_available():
        t0 = time.perf_counter()
        p = lstm_forecast(task.u, task.y, task.washout, task.n_train, cfg.lstm, seed=seed)
        rows.append(
            _row(task_name, "lstm", seed, cfg.lstm.hidden, "nrmse", nrmse(task.y[s:], p), t0)
        )
        if keep_predictions:
            preds["lstm"] = p

    if "linear_ar" in models:
        t0 = time.perf_counter()
        X = lagged(task.u, cfg.timeseries.ar_lags)
        model = train_ridge_readout(X[task.washout : s], task.y[task.washout : s])
        p = model.predict(X[s:])
        rows.append(_row(task_name, "linear_ar", seed, 0, "nrmse", nrmse(task.y[s:], p), t0))
        if keep_predictions:
            preds["linear_ar"] = p
    return rows, preds


# ---------------------------------------------------------------------------
# Memory capacity (Jaeger 2002)
# ---------------------------------------------------------------------------


def memory_curve(task: SeriesTask, states: NDArray[np.float32], max_delay: int) -> FloatArray:
    """Squared correlation between u[t-k] and its linear reconstruction, k = 1..max_delay."""
    X = np.asarray(states, dtype=np.float64)
    s = task.split
    curve = np.zeros(max_delay)
    for k in range(1, max_delay + 1):
        target = np.zeros_like(task.u)
        target[k:] = task.u[:-k]
        model = train_ridge_readout(X[task.washout : s], target[task.washout : s])
        pred = model.predict(X[s:])
        r = np.corrcoef(pred, target[s:])[0, 1]
        curve[k - 1] = 0.0 if not np.isfinite(r) else r**2
    return curve


def memory_seed(
    cfg: ExperimentConfig, seed: int, n_neurons: int | None = None
) -> tuple[list[Row], dict[str, FloatArray]]:
    """Memory capacity of the spiking reservoir and the ESN for one seed."""
    task = make_series_task(cfg, "memory", seed)
    fit_frames = task.u[: task.split, None]
    K = cfg.timeseries.memory_max_delay
    rows: list[Row] = []
    curves: dict[str, FloatArray] = {}
    builders: list[tuple[str, Callable[[], SpikingReservoir | EchoStateNetwork]]] = [
        (
            "spiking",
            lambda: build_spiking(
                cfg,
                build_encoder(cfg, 1, fit_frames, cfg.timeseries.electrodes_per_input),
                seed,
                n_neurons,
            ),
        ),
        ("esn", lambda: build_esn(cfg, build_encoder(cfg, 1, fit_frames), seed, n_neurons)),
    ]
    for name, build in builders:
        t0 = time.perf_counter()
        model = build()
        curve = memory_curve(task, series_states(model, task.u), K)
        curves[name] = curve
        rows.append(_row("memory", name, seed, model.n_neurons, "memory_capacity", curve.sum(), t0))
    return rows, curves
