"""3D views of the simulated organoid: neurons, electrodes and activity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from .plotting import INK, INK_2, SURFACE, _style  # noqa: E402
from .reservoir import SpikingReservoir  # noqa: E402

ELECTRODE_COLOR = "#eb6834"


def _setup_axes(ax: Any) -> None:
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_facecolor(SURFACE)


def plot_reservoir_3d(
    res: SpikingReservoir, activity: NDArray[np.float64], path: str | Path, n_edges: int = 250
) -> None:
    """Static 3D scatter: neurons coloured by firing rate, electrodes on the surface,
    and a sample of the strongest excitatory connections."""
    _style()
    fig = plt.figure(figsize=(9, 4.4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    _setup_axes(ax)
    p = res.positions
    pre, post = np.nonzero(res.mask & res.is_exc[:, None])
    if len(pre):
        idx = np.random.default_rng(0).choice(len(pre), min(n_edges, len(pre)), replace=False)
        for a, b in zip(pre[idx], post[idx], strict=True):
            ax.plot(*zip(p[a], p[b], strict=True), color=INK_2, lw=0.3, alpha=0.25)
    sc = ax.scatter(
        *p.T,
        c=activity,
        cmap="Blues",
        s=14,
        edgecolor=INK_2,
        lw=0.2,
        vmin=0,
        vmax=np.percentile(activity, 98) + 1e-9,
    )
    e = res.electrode_positions
    ax.scatter(*e.T, marker="s", s=28, color=ELECTRODE_COLOR, edgecolor=INK, lw=0.4)
    ax.set_title("Neurons (colour = firing rate) and electrodes", color=INK)
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02, label="rate (Hz)")

    ax2 = fig.add_subplot(1, 2, 2)
    d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    connected = d[res.mask]
    ax2.hist(
        d[np.triu_indices(len(p), 1)],
        bins=40,
        density=True,
        color=INK_2,
        alpha=0.35,
        label="all neuron pairs",
    )
    ax2.hist(connected, bins=40, density=True, color="#2a78d6", alpha=0.8, label="connected pairs")
    ax2.set_xlabel("3D distance (organoid radius = 1)")
    ax2.set_ylabel("density")
    ax2.set_title("Distance-dependent wiring")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def animate_reservoir(
    res: SpikingReservoir,
    spikes: NDArray[np.bool_],
    stim: NDArray[np.float64],
    frame_steps: int,
    path: str | Path,
    n_frames: int = 48,
    window: int = 10,
) -> None:
    """Rotating GIF of activity: each neuron lights up with its recent spike count and
    each electrode with its current stimulation intensity."""
    _style()
    steps = spikes.shape[0]
    times = np.linspace(window, steps - 1, n_frames).astype(int)
    fig = plt.figure(figsize=(4.0, 4.0), dpi=80)
    ax = fig.add_subplot(projection="3d")
    _setup_axes(ax)
    p = res.positions
    e = res.electrode_positions
    neurons = ax.scatter(
        *p.T, c=np.zeros(len(p)), cmap="Blues", vmin=0, vmax=2, s=10, edgecolor=INK_2, lw=0.15
    )
    elec = ax.scatter(
        *e.T,
        marker="s",
        s=22,
        c=np.zeros(len(e)),
        cmap="Oranges",
        vmin=0,
        vmax=1,
        edgecolor=INK,
        lw=0.3,
    )
    title = ax.set_title("", color=INK, fontsize=9)

    def update(k: int) -> Any:
        t = times[k]
        recent = spikes[t - window : t].sum(0).astype(float)
        neurons.set_array(recent)
        frame = min(t // frame_steps, stim.shape[0] - 1)
        elec.set_array(stim[frame])
        ax.view_init(elev=18, azim=30 + 300 * k / n_frames)
        title.set_text(f"t = {t} ms")
        return neurons, elec, title

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=10))
    plt.close(fig)
