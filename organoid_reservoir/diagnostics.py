"""Reservoir diagnostics.

* Firing statistics: mean rates of E and I cells, ISI coefficient of variation,
  population-rate burstiness and the spectral radius of the weight matrix.
* Kernel quality and generalisation rank (Legenstein & Maass 2007): the rank of
  the matrix of final states for many different input streams (higher is
  better) and for many noisy copies of one stream (lower is better).
* Separation property (Maass et al. 2002): how far apart the states of two
  input streams stay over time as a function of how different the inputs were.
"""

from __future__ import annotations

import copy

import numpy as np
from numpy.typing import NDArray

from .config import ExperimentConfig
from .encoding import ElectrodeEncoder
from .esn import EchoStateNetwork
from .reservoir import SpikingReservoir

FloatArray = NDArray[np.float64]


def firing_statistics(
    res: SpikingReservoir, stim: FloatArray, bin_ms: int = 10
) -> dict[str, float]:
    """Summary statistics of a run driven by ``stim`` (batch, frames, electrodes)."""
    out = res.run(stim, record_spikes=True)
    spikes = np.asarray(out["spikes"])  # (B, steps, n)
    B, steps, n = spikes.shape
    seconds = steps * res.config.dt / 1000.0
    rates = spikes.sum(1) / seconds  # (B, n)
    cvs = []
    for b in range(min(B, 4)):
        for i in range(n):
            t = np.flatnonzero(spikes[b, :, i])
            if len(t) >= 4:
                isi = np.diff(t)
                cvs.append(isi.std() / isi.mean())
    pop = spikes.sum(2)
    nb = steps // bin_ms
    binned: FloatArray = pop[:, : nb * bin_ms].reshape(B, nb, bin_ms).sum(2).astype(np.float64)
    burstiness = float(np.mean(binned.std(1) / (binned.mean(1) + 1e-9)))
    return {
        "rate_exc_hz": float(rates[:, res.is_exc].mean()),
        "rate_inh_hz": float(rates[:, ~res.is_exc].mean()),
        "isi_cv": float(np.mean(cvs)) if cvs else float("nan"),
        "pop_rate_cv": burstiness,
        "frac_silent": float((rates.mean(0) == 0).mean()),
        "spectral_radius": res.spectral_radius(),
    }


def effective_rank(states: FloatArray, energy: float = 0.99) -> int:
    """Number of principal components needed to capture ``energy`` of the variance."""
    X = np.asarray(states, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    if s.sum() == 0:
        return 0
    cum = np.cumsum(s**2) / np.sum(s**2)
    return int(np.searchsorted(cum, energy) + 1)


def _final_states(model: SpikingReservoir | EchoStateNetwork, u: FloatArray) -> FloatArray:
    states = model.transform(u)
    return np.asarray(states[:, -1], dtype=np.float64)


def kernel_and_generalization(
    model: SpikingReservoir | EchoStateNetwork,
    n_streams: int,
    n_frames: int,
    noise: float,
    seed: int = 0,
) -> tuple[int, int]:
    """Kernel quality and generalisation rank for 1-D random input streams in [-1, 1]."""
    rng = np.random.default_rng(seed)
    distinct = rng.uniform(-1, 1, (n_streams, n_frames, 1))
    base = rng.uniform(-1, 1, (1, n_frames, 1))
    noisy = base + noise * rng.standard_normal((n_streams, n_frames, 1))
    kq = effective_rank(_final_states(model, distinct))
    gr = effective_rank(_final_states(model, noisy))
    return kq, gr


def separation_curves(
    model: SpikingReservoir | EchoStateNetwork,
    distances: list[float],
    n_frames: int = 15,
    perturb_frames: int = 3,
    n_pairs: int = 16,
    seed: int = 0,
) -> dict[float, FloatArray]:
    """Mean normalised state distance over time for input pairs at several distances.

    Each pair shares the same stream except for the first ``perturb_frames``
    frames, where the second stream is shifted by ``d`` (in input units).
    The returned curves are divided by the mean state norm.
    """
    rng = np.random.default_rng(seed)
    curves: dict[float, FloatArray] = {}
    for d in distances:
        a = rng.uniform(-1, 1, (n_pairs, n_frames, 1))
        b = a.copy()
        signs = rng.choice([-1.0, 1.0], size=(n_pairs, perturb_frames, 1))
        b[:, :perturb_frames] += d * signs
        states = np.asarray(model.transform(np.concatenate([a, b])), dtype=np.float64)
        sa, sb = states[:n_pairs], states[n_pairs:]
        scale = np.linalg.norm(states, axis=2).mean() + 1e-12
        curves[d] = np.linalg.norm(sa - sb, axis=2).mean(0) / scale
    return curves


def scaled_models(
    cfg: ExperimentConfig,
    gain: float,
    encoder_spiking: ElectrodeEncoder,
    encoder_esn: ElectrodeEncoder,
    seed: int,
) -> tuple[SpikingReservoir, EchoStateNetwork]:
    """Spiking reservoir with all synaptic weights scaled by ``gain`` and an ESN whose
    spectral radius is scaled by the same factor."""
    rcfg = copy.deepcopy(cfg.reservoir)
    rcfg.w_exc *= gain
    rcfg.w_inh *= gain
    ecfg = copy.deepcopy(cfg.esn)
    ecfg.spectral_radius *= gain
    return (
        SpikingReservoir(rcfg, encoder_spiking, seed=seed),
        EchoStateNetwork(ecfg, encoder_esn, seed=seed),
    )
