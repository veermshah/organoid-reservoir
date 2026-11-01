"""Organoid-like spiking reservoir.

A vectorised NumPy network of Izhikevich (2003) neurons placed uniformly inside
a sphere. Connection probability falls off exponentially with 3D distance,
80% of cells are excitatory (regular-spiking / chattering mix) and 20% are
inhibitory (fast-spiking / low-threshold mix). Synaptic currents decay
exponentially and every neuron receives Gaussian background noise, which gives
the network spontaneous activity like a cultured organoid.

Optional pair-based STDP acts on excitatory-to-excitatory synapses. A
homeostatic rescaling step keeps each neuron's total excitatory input fixed,
so plasticity redistributes weights instead of driving runaway excitation.

The whole batch is simulated in parallel: state arrays have shape
(batch, n_neurons), and all batch members share the same synaptic weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .encoding import ElectrodeEncoder, fibonacci_sphere

FloatArray = NDArray[np.float64]
F32 = np.float32


@dataclass
class STDPConfig:
    """Pair-based STDP on E->E synapses with homeostatic rescaling."""

    a_plus: float = 0.01
    a_minus: float = 0.012
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    w_max_factor: float = 3.0
    """Weights are clipped to [0, w_max_factor * w_exc]."""
    normalize: bool = True


@dataclass
class ReservoirConfig:
    """Parameters of the spiking reservoir."""

    n_neurons: int = 400
    frac_exc: float = 0.8
    conn_lambda: float = 0.3
    """Length constant of the exponential connection-probability kernel (radius = 1)."""
    mean_in_degree: float = 40.0
    w_exc: float = 1.6
    w_inh: float = 4.0
    tau_syn: float = 5.0
    slow_frac: float = 0.0
    """Fraction of excitatory drive carried by a slow (NMDA-like) synaptic current."""
    tau_syn_slow: float = 100.0
    noise_exc: float = 3.0
    noise_inh: float = 2.0
    dt: float = 1.0
    trace_taus: list[float] = field(default_factory=lambda: [20.0])
    """Time constants of the exponential spike traces used as readout state. With
    several values the states of all traces are concatenated (filtered rates at
    several timescales)."""
    stdp: STDPConfig = field(default_factory=STDPConfig)


def _uniform_ball(rng: np.random.Generator, n: int) -> FloatArray:
    pts = rng.standard_normal((n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    r = rng.uniform(0.0, 1.0, n) ** (1.0 / 3.0)
    return pts * r[:, None]


class SpikingReservoir:
    """Izhikevich spiking network with 3D distance-dependent wiring.

    Parameters
    ----------
    config:
        Reservoir parameters.
    encoder:
        Electrode encoder that defines where stimulation enters the network.
    seed:
        Seed for wiring, cell parameters and simulation noise. Two reservoirs
        built with the same config, encoder and seed produce identical spikes.
    """

    def __init__(
        self, config: ReservoirConfig, encoder: ElectrodeEncoder, seed: int = 0
    ) -> None:
        self.config = config
        self.encoder = encoder
        self.seed = seed
        rng = np.random.default_rng(seed)
        n = config.n_neurons
        self.n_exc = int(round(config.frac_exc * n))
        self.is_exc = np.zeros(n, dtype=bool)
        self.is_exc[: self.n_exc] = True

        # Cell positions and Izhikevich parameters (Izhikevich 2003, heterogeneous).
        self.positions = _uniform_ball(rng, n)
        r = rng.uniform(0.0, 1.0, n)
        self.a = np.where(self.is_exc, 0.02, 0.02 + 0.08 * r)
        self.b = np.where(self.is_exc, 0.2, 0.25 - 0.05 * r)
        self.c = np.where(self.is_exc, -65.0 + 15.0 * r**2, -65.0)
        self.d = np.where(self.is_exc, 8.0 - 6.0 * r**2, 2.0)
        self.noise_sd = np.where(self.is_exc, config.noise_exc, config.noise_inh)

        # Distance-dependent connectivity, scaled to the requested mean in-degree.
        diff = self.positions[:, None, :] - self.positions[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        kernel = np.exp(-dist / config.conn_lambda)
        np.fill_diagonal(kernel, 0.0)
        scale = config.mean_in_degree * n / kernel.sum()
        prob = np.clip(kernel * scale, 0.0, 1.0)
        self.mask = rng.uniform(size=(n, n)) < prob  # mask[pre, post]
        mags = rng.uniform(0.5, 1.5, size=(n, n))
        signs = np.where(self.is_exc, config.w_exc, -config.w_inh)[:, None]
        self.W = (self.mask * mags * signs).astype(F32)
        self.W_init = self.W.copy()

        # Electrode coupling: Gaussian falloff with distance from each electrode.
        self.electrode_positions = fibonacci_sphere(encoder.n_electrodes, radius=1.0)
        ed = np.linalg.norm(
            self.electrode_positions[:, None, :] - self.positions[None, :, :], axis=-1
        )
        self.coupling = np.exp(-(ed**2) / (2 * encoder.config.spread**2)).astype(F32)

        self._sim_rng = np.random.default_rng(rng.integers(2**63))

    # ------------------------------------------------------------------
    @property
    def n_neurons(self) -> int:
        return self.config.n_neurons

    def reset_weights(self) -> None:
        """Restore the weights present at construction (undo plasticity)."""
        self.W = self.W_init.copy()

    def reseed_noise(self, seed: int) -> None:
        """Reset the simulation-noise generator (wiring is unchanged)."""
        self._sim_rng = np.random.default_rng(seed)

    def spectral_radius(self) -> float:
        """Largest absolute eigenvalue of the current weight matrix."""
        return float(np.max(np.abs(np.linalg.eigvals(self.W.astype(np.float64)))))

    # ------------------------------------------------------------------
    def run(
        self,
        stim: FloatArray,
        lengths: NDArray[np.int64] | None = None,
        plastic: bool = False,
        record_spikes: bool = False,
        warmup_steps: int = 50,
    ) -> dict[str, NDArray[np.float32] | NDArray[np.bool_]]:
        """Simulate the network driven by electrode intensities.

        Parameters
        ----------
        stim:
            Electrode intensities in [0, 1], shape (batch, n_frames, n_electrodes).
            Each frame is held for ``encoder.config.frame_steps`` simulation steps.
        lengths:
            Valid number of frames per batch element; stimulation stops after it.
        plastic:
            Apply STDP to E->E synapses during the run.
        record_spikes:
            Also return the full boolean spike raster (batch, steps, n_neurons).
        warmup_steps:
            Unstimulated steps run first so the network settles into its
            spontaneous state.

        Returns
        -------
        dict with ``states`` (batch, n_frames, n * len(trace_taus)) holding the
        exponential spike traces at the end of each frame, ``counts`` (batch, n_frames, n) holding
        spike counts per frame and optionally ``spikes``.
        """
        cfg = self.config
        ecfg = self.encoder.config
        rng = self._sim_rng
        B, T, E = stim.shape
        n = cfg.n_neurons
        steps = ecfg.frame_steps
        if lengths is None:
            lengths = np.full(B, T, dtype=np.int64)

        v = np.full((B, n), -65.0, dtype=F32)
        u = (self.b * v).astype(F32)
        a, b, c, d = (x.astype(F32) for x in (self.a, self.b, self.c, self.d))
        noise_sd = self.noise_sd.astype(F32)
        i_syn = np.zeros((B, n), dtype=F32)
        i_slow = np.zeros((B, n), dtype=F32)
        n_tr = len(cfg.trace_taus)
        trace = np.zeros((B, n_tr, n), dtype=F32)
        syn_decay = F32(np.exp(-cfg.dt / cfg.tau_syn))
        slow_decay = F32(np.exp(-cfg.dt / cfg.tau_syn_slow))
        trace_decay = np.exp(-cfg.dt / np.asarray(cfg.trace_taus, dtype=F32))[None, :, None]
        trace_decay = trace_decay.astype(F32)
        slow_frac = F32(cfg.slow_frac)
        # Slow component is normalised by its time constant so total charge is preserved.
        slow_gain = F32(cfg.tau_syn / cfg.tau_syn_slow)

        ne = self.n_exc
        st = cfg.stdp
        w_max = F32(st.w_max_factor * cfg.w_exc)
        mask_ee = self.mask[:ne, :ne]
        target_in = np.abs(self.W_init[:ne, :ne]).sum(0) if st.normalize else None
        x_pre = np.zeros((B, ne), dtype=F32)
        x_post = np.zeros((B, ne), dtype=F32)
        dec_p = F32(np.exp(-cfg.dt / st.tau_plus))
        dec_m = F32(np.exp(-cfg.dt / st.tau_minus))

        states = np.zeros((B, T, n_tr * n), dtype=F32)
        counts = np.zeros((B, T, n), dtype=F32)
        raster = np.zeros((B, T * steps, n), dtype=bool) if record_spikes else None

        if ecfg.mode == "pulse":
            p_pulse = stim * (ecfg.max_rate_hz * cfg.dt / 1000.0)
        elif ecfg.mode == "current":
            dc = (stim.astype(F32) @ self.coupling) * F32(ecfg.current_gain)
        else:
            raise ValueError(f"unknown stimulation mode {ecfg.mode!r}")
        active = (np.arange(T)[None, :] < lengths[:, None]).astype(F32)  # (B, T)

        spikes = np.zeros((B, n), dtype=bool)
        total_steps = warmup_steps + T * steps
        for k in range(total_steps):
            frame = (k - warmup_steps) // steps
            stimulating = k >= warmup_steps
            # synaptic input from last step's spikes
            i_syn *= syn_decay
            if spikes.any():
                syn_in = spikes.astype(F32) @ self.W
                if slow_frac > 0:
                    exc_in = spikes[:, :ne].astype(F32) @ self.W[:ne]
                    syn_in -= slow_frac * exc_in
                    i_slow *= slow_decay
                    i_slow += slow_frac * slow_gain * exc_in
                i_syn += syn_in
            elif slow_frac > 0:
                i_slow *= slow_decay
            current = i_syn + i_slow + noise_sd * rng.standard_normal((B, n), dtype=F32)
            if stimulating:
                gate = active[:, frame][:, None]
                if ecfg.mode == "pulse":
                    pulses = rng.uniform(size=(B, E)) < p_pulse[:, frame]
                    if pulses.any():
                        current += (pulses.astype(F32) @ self.coupling) * (
                            F32(ecfg.pulse_amplitude) * gate
                        )
                else:
                    current += dc[:, frame] * gate
            # Izhikevich update, two half steps for v as in the original model
            for _ in range(2):
                v += F32(0.5) * (F32(0.04) * v * v + F32(5.0) * v + F32(140.0) - u + current)
                np.minimum(v, F32(35.0), out=v)
            u += a * (b * v - u)
            spikes = v >= 30.0
            if spikes.any():
                v = np.where(spikes, c, v)
                u = u + spikes * d
            trace *= trace_decay
            trace += spikes[:, None, :]

            if plastic:
                s_e = spikes[:, :ne].astype(F32)
                x_pre *= dec_p
                x_post *= dec_m
                if s_e.any():
                    ltp = x_pre.T @ s_e  # pre trace x post spike
                    ltd = s_e.T @ x_post  # pre spike x post trace
                    dw = (F32(st.a_plus) * ltp - F32(st.a_minus) * ltd) / F32(B)
                    w_ee = self.W[:ne, :ne]
                    w_ee += dw * mask_ee * F32(cfg.w_exc)
                    np.clip(w_ee, 0.0, w_max, out=w_ee)
                x_pre += s_e
                x_post += s_e
                if target_in is not None and (k + 1) % 100 == 0:
                    w_ee = self.W[:ne, :ne]
                    col = w_ee.sum(0)
                    w_ee *= np.where(col > 0, target_in / np.maximum(col, 1e-6), 1.0).astype(F32)

            if stimulating:
                counts[:, frame] += spikes
                if raster is not None:
                    raster[:, k - warmup_steps] = spikes
                if (k - warmup_steps + 1) % steps == 0:
                    states[:, frame] = trace.reshape(B, -1)

        out: dict[str, NDArray[np.float32] | NDArray[np.bool_]] = {
            "states": states,
            "counts": counts,
        }
        if raster is not None:
            out["spikes"] = raster
        return out

    def transform(
        self,
        inputs: FloatArray,
        lengths: NDArray[np.int64] | None = None,
        plastic: bool = False,
        batch_size: int = 128,
    ) -> NDArray[np.float32]:
        """Encode raw inputs (batch, n_frames, n_inputs) and return per-frame states."""
        stim = self.encoder.intensities(inputs)
        B = stim.shape[0]
        if lengths is None:
            lengths = np.full(B, stim.shape[1], dtype=np.int64)
        chunks = []
        for i in range(0, B, batch_size):
            res = self.run(stim[i : i + batch_size], lengths[i : i + batch_size], plastic)
            chunks.append(np.asarray(res["states"]))
        return np.concatenate(chunks, 0)
