"""Echo state network baseline (Jaeger 2001), batched and leaky."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .encoding import ElectrodeEncoder

FloatArray = NDArray[np.float64]


@dataclass
class ESNConfig:
    """Parameters of the tanh echo state network."""

    n_neurons: int = 400
    spectral_radius: float = 0.9
    leak_rate: float = 0.5
    input_scaling: float = 0.5
    bias_scaling: float = 0.1
    density: float = 0.1


class EchoStateNetwork:
    """Leaky tanh reservoir with sparse random recurrent weights.

    It reads the same z-scored inputs as the spiking reservoir (through the
    encoder's normaliser, not the electrode code) and updates once per frame.
    """

    def __init__(self, config: ESNConfig, encoder: ElectrodeEncoder, seed: int = 0) -> None:
        self.config = config
        self.encoder = encoder
        rng = np.random.default_rng(seed)
        n = config.n_neurons
        mask = rng.uniform(size=(n, n)) < config.density
        W = rng.standard_normal((n, n)) * mask
        rho = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = (W * (config.spectral_radius / rho)).astype(np.float32)
        d = encoder.n_inputs
        self.W_in = (rng.uniform(-1, 1, (d, n)) * config.input_scaling).astype(np.float32)
        self.bias = (rng.uniform(-1, 1, n) * config.bias_scaling).astype(np.float32)

    @property
    def n_neurons(self) -> int:
        return self.config.n_neurons

    def transform(
        self, inputs: FloatArray, lengths: NDArray[np.int64] | None = None
    ) -> NDArray[np.float32]:
        """Return states (batch, n_frames, n) for raw inputs (batch, n_frames, n_inputs)."""
        z = self.encoder.normalize(inputs).astype(np.float32)
        B, T, _ = z.shape
        if lengths is None:
            lengths = np.full(B, T, dtype=np.int64)
        alpha = np.float32(self.config.leak_rate)
        x = np.zeros((B, self.n_neurons), dtype=np.float32)
        states = np.zeros((B, T, self.n_neurons), dtype=np.float32)
        drive = z @ self.W_in  # (B, T, n)
        for t in range(T):
            x_new = np.tanh(x @ self.W + drive[:, t] + self.bias)
            x = (1 - alpha) * x + alpha * x_new
            states[:, t] = x
        return states
