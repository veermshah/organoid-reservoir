"""Turn per-frame reservoir states into fixed-size readout features."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def pool_sequence_states(
    states: NDArray[np.floating], lengths: NDArray[np.int64], n_segments: int = 3
) -> NDArray[np.float64]:
    """Pool (batch, n_frames, n) states over each sequence's valid frames.

    The feature vector is the mean over the whole utterance followed by the mean
    over ``n_segments`` equal-length temporal segments, giving
    ``(1 + n_segments) * n`` features. This keeps coarse temporal order while
    handling variable-length inputs.
    """
    B, _, n = states.shape
    feats = np.zeros((B, (1 + n_segments) * n))
    for i in range(B):
        L = int(lengths[i])
        s = np.asarray(states[i, :L], dtype=np.float64)
        parts = [s.mean(0)]
        edges = np.linspace(0, L, n_segments + 1)
        for j in range(n_segments):
            lo = int(np.floor(edges[j]))
            hi = max(int(np.ceil(edges[j + 1])), lo + 1)
            parts.append(s[lo:hi].mean(0))
        feats[i] = np.concatenate(parts)
    return feats


def resample_sequences(seqs: list[NDArray[np.float64]], n_frames: int) -> NDArray[np.float64]:
    """Linearly resample variable-length (T_i, d) sequences to (B, n_frames * d)."""
    out = []
    for s in seqs:
        t_old = np.linspace(0.0, 1.0, len(s))
        t_new = np.linspace(0.0, 1.0, n_frames)
        out.append(np.stack([np.interp(t_new, t_old, s[:, j]) for j in range(s.shape[1])], 1))
    return np.stack(out).reshape(len(seqs), -1)


def mean_std_features(seqs: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Time-averaged features: per-coefficient mean and standard deviation."""
    return np.stack([np.concatenate([s.mean(0), s.std(0)]) for s in seqs])
