"""Virtual electrode array that turns input features into spatiotemporal stimulation.

Electrodes sit on the surface of the simulated organoid. Each input feature is
assigned ``electrodes_per_input`` electrodes with alternating polarity and
staggered thresholds, so a positive value drives one group, a negative value
drives another, and large values recruit more electrodes. The electrode
intensity (0..1) is then delivered to nearby neurons either as Poisson pulse
trains (default, closer to how an MEA stimulates tissue) or as a DC current.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass
class EncoderConfig:
    """Parameters of the virtual electrode array."""

    electrodes_per_input: int = 2
    max_threshold: float = 1.5
    """Largest threshold (in z-scored units) among electrodes of the same polarity."""
    spread: float = 0.35
    """Gaussian radius of the electrode-to-neuron coupling (organoid radius = 1)."""
    mode: str = "pulse"
    """``pulse`` (Poisson pulse trains) or ``current`` (DC injection)."""
    max_rate_hz: float = 200.0
    pulse_amplitude: float = 40.0
    current_gain: float = 20.0
    frame_steps: int = 20
    """Simulation steps (ms) each input frame is held for."""


def fibonacci_sphere(n: int, radius: float = 1.0) -> FloatArray:
    """Roughly uniform points on a sphere surface; consecutive points are neighbours."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    pts = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
    return radius * pts


class ElectrodeEncoder:
    """Maps a (batch, time, n_inputs) feature tensor to electrode intensities in [0, 1].

    Features are z-scored with statistics from :meth:`fit`, so the encoder can be
    shared between the spiking reservoir and the echo state network baseline.
    """

    def __init__(self, n_inputs: int, config: EncoderConfig | None = None) -> None:
        self.n_inputs = n_inputs
        self.config = config or EncoderConfig()
        k = self.config.electrodes_per_input
        if k < 2 or k % 2:
            raise ValueError("electrodes_per_input must be an even number >= 2")
        half = k // 2
        self.polarity = np.tile(np.array([1.0, -1.0]), half)  # (k,)
        self.thresholds = np.repeat(np.linspace(0.0, self.config.max_threshold, half), 2)
        if half == 1:
            self.thresholds = np.zeros(2)
        self.mean = np.zeros(n_inputs)
        self.std = np.ones(n_inputs)

    @property
    def n_electrodes(self) -> int:
        return self.n_inputs * self.config.electrodes_per_input

    def fit(self, frames: FloatArray) -> ElectrodeEncoder:
        """Estimate per-feature mean/std from a (n_frames, n_inputs) array."""
        frames = np.asarray(frames, dtype=np.float64).reshape(-1, self.n_inputs)
        self.mean = frames.mean(0)
        self.std = frames.std(0) + 1e-8
        return self

    def normalize(self, u: FloatArray) -> FloatArray:
        return (np.asarray(u, dtype=np.float64) - self.mean) / self.std

    def intensities(self, u: FloatArray) -> FloatArray:
        """Return electrode intensities with shape ``u.shape[:-1] + (n_electrodes,)``."""
        z = self.normalize(u)[..., :, None]  # (..., n_inputs, 1)
        drive = np.maximum(self.polarity * z - self.thresholds, 0.0)
        s = np.tanh(drive)  # (..., n_inputs, k)
        return s.reshape(*s.shape[:-2], self.n_electrodes)


def pad_sequences(seqs: list[FloatArray]) -> tuple[FloatArray, NDArray[np.int64]]:
    """Zero-pad a list of (T_i, d) arrays to (B, T_max, d); return the array and lengths."""
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    d = seqs[0].shape[1]
    out = np.zeros((len(seqs), int(lengths.max()), d))
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return out, lengths
