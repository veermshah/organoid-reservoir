"""Figures for the README and report. Every function writes one PNG."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

# Fixed colour per model, so a model keeps its colour in every figure.
MODEL_COLORS = {
    "spiking": "#2a78d6",
    "esn": "#eb6834",
    "lstm": "#1baf7a",
    "spiking_dc": "#eda100",
    "spiking_stdp": "#e87ba4",
    "logreg_mean_std": "#008300",
    "logreg_resampled": "#6250d6",
    "linear_ar": "#e34948",
}
MODEL_LABELS = {
    "spiking": "Spiking reservoir (pulse)",
    "spiking_stdp": "Spiking + STDP (pulse)",
    "spiking_dc": "Spiking reservoir (DC)",
    "esn": "Echo state network",
    "lstm": "LSTM",
    "logreg_mean_std": "LogReg, mean+std",
    "logreg_resampled": "LogReg, resampled",
    "linear_ar": "Linear AR (10 lags)",
}
TASK_LABELS = {"henon": "Hénon map", "narma10": "NARMA-10", "mackey_glass": "Mackey-Glass"}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": INK_2,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "legend.frameon": False,
            "font.size": 9,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _seed_bars(ax: plt.Axes, df: pd.DataFrame, order: list[str], log: bool = False) -> None:
    """Bars at the seed mean, ±1 std whiskers, and one dot per seed."""
    rng = np.random.default_rng(0)
    for i, model in enumerate(order):
        vals = df.loc[df.model == model, "value"].to_numpy()
        if len(vals) == 0:
            continue
        color = MODEL_COLORS.get(model, INK_2)
        ax.bar(i, vals.mean(), width=0.62, color=color, alpha=0.85, edgecolor=SURFACE, lw=2)
        ax.errorbar(i, vals.mean(), yerr=vals.std(), color=INK, capsize=3, lw=1)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals, s=10, color=INK, zorder=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in order], rotation=25, ha="right")
    if log:
        ax.set_yscale("log")


def plot_vowels(df: pd.DataFrame, path: str | Path) -> None:
    _style()
    acc = df[df.metric == "accuracy"]
    order = [m for m in MODEL_LABELS if m in set(acc.model)]
    order.sort(key=lambda m: -acc.loc[acc.model == m, "value"].mean())
    fig, ax = plt.subplots(figsize=(7, 3.6))
    _seed_bars(ax, acc, order)
    lo = max(0.0, acc.value.min() - 0.05)
    ax.set_ylim(lo, 1.0)
    ax.set_ylabel("Test accuracy (370 utterances)")
    ax.set_title("Japanese Vowels speaker recognition (9 speakers)")
    _save(fig, path)


def plot_timeseries(df: pd.DataFrame, path: str | Path) -> None:
    _style()
    err = df[df.metric == "nrmse"]
    tasks = [t for t in TASK_LABELS if t in set(err.task)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks), 3.6), squeeze=False)
    for ax, task in zip(axes[0], tasks, strict=True):
        sub = err[err.task == task]
        order = [m for m in MODEL_LABELS if m in set(sub.model)]
        order.sort(key=lambda m: sub.loc[sub.model == m, "value"].mean())
        _seed_bars(ax, sub, order, log=True)
        ax.set_title(TASK_LABELS[task])
    axes[0][0].set_ylabel("Test NRMSE (log scale, lower is better)")
    fig.tight_layout()
    _save(fig, path)


def plot_predictions(preds: dict[str, NDArray[np.float64]], path: str | Path, n: int = 80) -> None:
    _style()
    models = [m for m in ("spiking", "spiking_dc", "esn", "lstm") if m in preds]
    fig, axes = plt.subplots(len(models), 1, figsize=(7, 1.5 * len(models) + 0.6), sharex=True)
    axes = np.atleast_1d(axes)
    t = np.arange(n)
    for ax, m in zip(axes, models, strict=True):
        ax.plot(t, preds["target"][:n], color=INK_2, lw=1.2, label="target")
        ax.plot(t, preds[m][:n], color=MODEL_COLORS[m], lw=2, label=MODEL_LABELS[m])
        ax.legend(loc="upper right", ncol=2, fontsize=8)
        ax.set_ylabel("x")
    axes[-1].set_xlabel("Test step")
    axes[0].set_title("Hénon map one-step prediction (first 80 test steps, seed 0)")
    fig.tight_layout()
    _save(fig, path)


def plot_size_sweep(df: pd.DataFrame, path: str | Path) -> None:
    _style()
    panels = [("vowels", "accuracy", "Vowels accuracy"), ("henon", "nrmse", "Hénon NRMSE")]
    panels = [p for p in panels if p[0] in set(df.task)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4), squeeze=False)
    for ax, (task, metric, label) in zip(axes[0], panels, strict=True):
        sub = df[(df.task == task) & (df.metric == metric)]
        for model in ("spiking", "spiking_dc", "esn"):
            g = sub[sub.model == model].groupby("n_neurons").value
            if g.ngroups == 0:
                continue
            m, s = g.mean(), g.std().fillna(0)
            c = MODEL_COLORS[model]
            ax.plot(m.index, m.values, "-o", color=c, lw=2, ms=6, label=MODEL_LABELS[model])
            ax.fill_between(m.index, m - s, m + s, color=c, alpha=0.15, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(sub.n_neurons.unique()))
        ax.set_xticklabels(sorted(sub.n_neurons.unique()))
        ax.set_xlabel("Reservoir size (neurons)")
        ax.set_title(label)
        if metric == "nrmse":
            ax.set_yscale("log")
        ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def plot_stdp(df: pd.DataFrame, path: str | Path) -> None:
    """Paired per-seed comparison of the pulse reservoir before and after STDP exposure."""
    _style()
    rows = []
    for task, metric in (
        ("vowels", "accuracy"),
        ("henon", "nrmse"),
        ("narma10", "nrmse"),
        ("mackey_glass", "nrmse"),
    ):
        sub = df[(df.task == task) & (df.metric == metric)]
        a = sub[sub.model == "spiking"].set_index("seed").value
        b = sub[sub.model == "spiking_stdp"].set_index("seed").value
        common = a.index.intersection(b.index)
        if len(common):
            rows.append((task, metric, a[common], b[common]))
    fig, axes = plt.subplots(1, len(rows), figsize=(2.4 * len(rows) + 0.5, 3.2), squeeze=False)
    for ax, (task, metric, a, b) in zip(axes[0], rows, strict=True):
        for s in a.index:
            ax.plot([0, 1], [a[s], b[s]], color=INK_2, lw=1, alpha=0.6)
        ax.scatter(np.zeros(len(a)), a, color=MODEL_COLORS["spiking"], s=30, zorder=3)
        ax.scatter(np.ones(len(b)), b, color=MODEL_COLORS["spiking_stdp"], s=30, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["naive", "after STDP"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(TASK_LABELS.get(task, "Vowels"))
        ax.set_ylabel(
            "accuracy (higher is better)" if metric == "accuracy" else "NRMSE (lower is better)"
        )
    fig.suptitle("Unsupervised STDP exposure, paired by seed", color=INK, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def plot_memory(curves: pd.DataFrame, path: str | Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for model in ("spiking", "esn"):
        sub = curves[curves.model == model]
        if sub.empty:
            continue
        g = sub.groupby("delay").r2
        m, s = g.mean(), g.std().fillna(0)
        total = sub.groupby("seed").r2.sum()
        c = MODEL_COLORS[model]
        label = f"{MODEL_LABELS[model]} (MC = {total.mean():.2f} ± {total.std():.2f})"
        ax.plot(m.index, m.to_numpy(), "-", color=c, lw=2, label=label)
        ax.fill_between(m.index, m - s, m + s, color=c, alpha=0.15, lw=0)
    ax.set_xlabel("Delay k (frames)")
    ax.set_ylabel("r² of reconstructed u[t-k]")
    ax.set_title("Linear memory capacity")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_kernel_quality(df: pd.DataFrame, path: str | Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for ax, model in zip(axes, ("spiking", "esn"), strict=True):
        sub = df[df.model == model].groupby("gain")[["kernel_quality", "generalization_rank"]]
        m = sub.mean()
        c = MODEL_COLORS[model]
        ax.plot(m.index, m.kernel_quality, "-o", color=c, lw=2, ms=6, label="kernel quality")
        ax.plot(
            m.index,
            m.generalization_rank,
            "--s",
            color=INK_2,
            lw=2,
            ms=6,
            label="generalisation rank",
        )
        ax.set_xlabel("Weight gain (x default)")
        ax.set_title(MODEL_LABELS[model])
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Effective rank of final states")
    fig.suptitle(
        "Kernel quality vs generalisation rank (Legenstein & Maass 2007)",
        color=INK,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, path)


def plot_separation(curves: dict[str, dict[float, NDArray[np.float64]]], path: str | Path) -> None:
    _style()
    fig, axes = plt.subplots(1, len(curves), figsize=(4.2 * len(curves), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    blues = ["#cfe0f6", "#8ab4ea", "#3d84d9", "#1c4f8f"]
    for ax, (model, cv) in zip(axes, curves.items(), strict=True):
        for color, (d, y) in zip(blues, sorted(cv.items()), strict=False):
            ax.plot(
                np.arange(1, len(y) + 1), y, "-", color=color, lw=2, label=f"input distance {d:g}"
            )
        ax.axvspan(0.5, 3.5, color=GRID, alpha=0.5, lw=0)
        ax.set_xlabel("Frame (inputs differ only in frames 1-3)")
        ax.set_title(MODEL_LABELS[model])
        ax.legend(fontsize=7)
    axes[0].set_ylabel("State distance / mean state norm")
    fig.suptitle("Separation property", color=INK, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def plot_noise_ablation(df: pd.DataFrame, path: str | Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, (metric, label) in zip(
        axes,
        (("henon_nrmse", "Hénon NRMSE"), ("memory_capacity", "Memory capacity")),
        strict=True,
    ):
        g = df.groupby("noise_scale")[metric]
        m, s = g.mean(), g.std().fillna(0)
        c = MODEL_COLORS["spiking_dc"]
        ax.errorbar(m.index, m.values, yerr=s.values, fmt="-o", color=c, lw=2, ms=6, capsize=3)
        ax.set_xlabel("Background noise (x default)")
        ax.set_title(label + " of the DC spiking reservoir")
    fig.tight_layout()
    _save(fig, path)


def plot_hyper_sweep(
    grid: NDArray[np.float64],
    x_values: list[float],
    y_values: list[float],
    x_label: str,
    y_label: str,
    title: str,
    path: str | Path,
) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(grid, origin="lower", cmap="Blues", aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([f"{v:g}" for v in x_values])
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels([f"{v:g}" for v in y_values])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    mid = np.nanmin(grid) + 0.6 * (np.nanmax(grid) - np.nanmin(grid))
    for (i, j), v in np.ndenumerate(grid):
        ax.text(
            j,
            i,
            f"{v:.3f}",
            ha="center",
            va="center",
            fontsize=8,
            color=SURFACE if v > mid else INK,
        )
    fig.colorbar(im, ax=ax, shrink=0.85, label="mean test accuracy")
    ax.set_title(title)
    _save(fig, path)


def plot_raster(
    spikes: NDArray[np.bool_],
    stim: NDArray[np.float64],
    is_exc: NDArray[np.bool_],
    frame_steps: int,
    path: str | Path,
    title: str,
) -> None:
    """Stimulation intensities, spike raster and population rate for one input."""
    _style()
    fig, axes = plt.subplots(
        3, 1, figsize=(7, 5.4), sharex=True, gridspec_kw={"height_ratios": [1, 3, 1]}
    )
    steps = spikes.shape[0]
    axes[0].imshow(
        np.repeat(stim.T, frame_steps, axis=1),
        aspect="auto",
        cmap="Blues",
        extent=(0, steps, stim.shape[1], 0),
        interpolation="nearest",
    )
    axes[0].grid(False)
    axes[0].set_ylabel("electrode")
    axes[0].set_title(title)
    t, i = np.nonzero(spikes)
    exc = is_exc[i]
    axes[1].scatter(t[exc], i[exc], s=1.2, color=MODEL_COLORS["spiking"], lw=0, label="excitatory")
    axes[1].scatter(t[~exc], i[~exc], s=1.2, color=MODEL_COLORS["esn"], lw=0, label="inhibitory")
    axes[1].set_ylabel("neuron")
    axes[1].set_ylim(spikes.shape[1], 0)
    axes[1].legend(loc="upper right", markerscale=6, fontsize=8)
    rate = spikes.sum(1).astype(float)
    kernel = np.ones(5) / 5
    axes[2].plot(np.convolve(rate, kernel, mode="same") / spikes.shape[1] * 1000, color=INK, lw=1.2)
    axes[2].set_ylabel("pop. rate (Hz)")
    axes[2].set_xlabel("time (ms)")
    fig.tight_layout()
    _save(fig, path)
