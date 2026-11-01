"""Datasets used in the experiments.

* Japanese Vowels (UCI #128): nine male speakers saying the vowel pair /ae/,
  each utterance a variable-length sequence of 12 LPC cepstrum coefficients.
  Downloaded on demand into ``data/raw`` and never committed.
* Synthetic time series: the Henon map, NARMA-10, Mackey-Glass and i.i.d.
  input for the memory-capacity measurement.
"""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/JapaneseVowels-mld/"
UCI_ZIP = "https://archive.ics.uci.edu/static/public/128/japanese+vowels.zip"
VOWEL_FILES = ("ae.train", "ae.test")
SHA256 = {
    "ae.train": "a5976f6ae14ffed6ae996b528ad5c0c1e1a7984585feb43ebcbd1b8a09d38ab2",
    "ae.test": "c4c61d2fcd3fa6054978bb4e7a3ff38e96d7af41912101b75fde875a5b2432a6",
}
# Number of utterances per speaker, in file order (from size_ae.train / size_ae.test).
TRAIN_SIZES = (30, 30, 30, 30, 30, 30, 30, 30, 30)
TEST_SIZES = (31, 35, 88, 44, 29, 24, 40, 50, 29)
N_COEFFS = 12


@dataclass
class SequenceDataset:
    """Variable-length multivariate sequences with integer class labels."""

    X_train: list[FloatArray]
    y_train: NDArray[np.int64]
    X_test: list[FloatArray]
    y_test: NDArray[np.int64]

    @property
    def n_classes(self) -> int:
        return int(max(self.y_train.max(), self.y_test.max()) + 1)

    @property
    def n_features(self) -> int:
        return int(self.X_train[0].shape[1])


def parse_vowel_blocks(text: str) -> list[FloatArray]:
    """Parse the UCI ``ae.*`` format: blocks of 12-float rows separated by blank lines."""
    blocks: list[FloatArray] = []
    rows: list[list[float]] = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            if rows:
                blocks.append(np.asarray(rows, dtype=np.float64))
                rows = []
            continue
        rows.append([float(p) for p in parts])
    if rows:
        blocks.append(np.asarray(rows, dtype=np.float64))
    return blocks


def labels_from_sizes(sizes: tuple[int, ...] | list[int]) -> NDArray[np.int64]:
    """Expand per-speaker counts into a label vector (speakers are stored in order)."""
    return np.repeat(np.arange(len(sizes), dtype=np.int64), sizes)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_japanese_vowels(data_dir: str | Path = "data/raw", timeout: float = 30.0) -> Path:
    """Download ``ae.train`` and ``ae.test`` into ``data_dir/japanese_vowels``.

    Tries the classic UCI directory first and falls back to the zip archive on
    the new UCI site. Files are checked against known SHA-256 hashes.
    """
    target = Path(data_dir) / "japanese_vowels"
    target.mkdir(parents=True, exist_ok=True)
    if all((target / f).exists() and _sha256(target / f) == SHA256[f] for f in VOWEL_FILES):
        return target
    try:
        for name in VOWEL_FILES:
            with urllib.request.urlopen(UCI_BASE + name, timeout=timeout) as resp:
                (target / name).write_bytes(resp.read())
    except OSError:
        with urllib.request.urlopen(UCI_ZIP, timeout=timeout) as resp:
            archive = zipfile.ZipFile(BytesIO(resp.read()))
        for name in VOWEL_FILES:
            member = next(m for m in archive.namelist() if m.endswith(name))
            (target / name).write_bytes(archive.read(member))
    for name in VOWEL_FILES:
        digest = _sha256(target / name)
        if digest != SHA256[name]:
            raise RuntimeError(f"checksum mismatch for {name}: {digest}")
    return target


def load_japanese_vowels(
    data_dir: str | Path = "data/raw", download: bool = True
) -> SequenceDataset:
    """Load the Japanese Vowels train/test split (270 / 370 utterances, 9 speakers)."""
    folder = Path(data_dir) / "japanese_vowels"
    if download:
        folder = download_japanese_vowels(data_dir)
    return load_vowel_files(folder / "ae.train", folder / "ae.test", TRAIN_SIZES, TEST_SIZES)


def load_vowel_files(
    train_path: str | Path,
    test_path: str | Path,
    train_sizes: tuple[int, ...] | list[int],
    test_sizes: tuple[int, ...] | list[int],
) -> SequenceDataset:
    """Load a pair of files in UCI ``ae.*`` format with the given per-speaker counts."""
    X_train = parse_vowel_blocks(Path(train_path).read_text())
    X_test = parse_vowel_blocks(Path(test_path).read_text())
    y_train = labels_from_sizes(train_sizes)
    y_test = labels_from_sizes(test_sizes)
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError(
            f"block count mismatch: {len(X_train)}/{len(y_train)} train, "
            f"{len(X_test)}/{len(y_test)} test"
        )
    return SequenceDataset(X_train, y_train, X_test, y_test)


# ---------------------------------------------------------------------------
# Synthetic time series
# ---------------------------------------------------------------------------


def henon_map(
    n: int, a: float = 1.4, b: float = 0.3, x0: float = 0.1, y0: float = 0.1, discard: int = 100
) -> FloatArray:
    """Return ``n`` samples of the x coordinate of the Henon map.

    x[t+1] = 1 - a x[t]^2 + y[t],  y[t+1] = b x[t]
    """
    x, y = x0, y0
    out = np.empty(n + discard)
    for t in range(n + discard):
        x, y = 1.0 - a * x * x + y, b * x
        out[t] = x
    return out[discard:]


def narma10(n: int, seed: int = 0, discard: int = 200) -> tuple[FloatArray, FloatArray]:
    """NARMA-10 benchmark. Returns input ``u`` ~ U(0, 0.5) and target ``y``.

    y[t+1] = 0.3 y[t] + 0.05 y[t] sum_{i=0}^{9} y[t-i] + 1.5 u[t-9] u[t] + 0.1
    The model sees u[t] and must output y[t+1].
    """
    rng = np.random.default_rng(seed)
    total = n + discard
    u = rng.uniform(0.0, 0.5, total)
    y = np.zeros(total + 1)
    for t in range(9, total):
        y[t + 1] = (
            0.3 * y[t] + 0.05 * y[t] * y[t - 9 : t + 1].sum() + 1.5 * u[t - 9] * u[t] + 0.1
        )
    return u[discard:], y[discard + 1 :]


def mackey_glass(
    n: int,
    tau: float = 17.0,
    beta: float = 0.2,
    gamma: float = 0.1,
    power: float = 10.0,
    dt: float = 0.1,
    sample_every: float = 3.0,
    discard: int = 300,
    seed: int = 0,
) -> FloatArray:
    """Mackey-Glass delay differential equation, integrated with RK4 and subsampled.

    dx/dt = beta x(t - tau) / (1 + x(t - tau)^power) - gamma x(t)
    """
    rng = np.random.default_rng(seed)
    hist_len = int(round(tau / dt))
    stride = int(round(sample_every / dt))
    steps = (n + discard) * stride
    x = np.empty(steps + hist_len + 1)
    x[: hist_len + 1] = 1.2 + 0.01 * rng.standard_normal(hist_len + 1)

    def f(xt: float, xlag: float) -> float:
        return beta * xlag / (1.0 + xlag**power) - gamma * xt

    for i in range(hist_len, hist_len + steps):
        lag = x[i - hist_len]
        lag_next = x[i - hist_len + 1]
        lag_mid = 0.5 * (lag + lag_next)
        xi = x[i]
        k1 = f(xi, lag)
        k2 = f(xi + 0.5 * dt * k1, lag_mid)
        k3 = f(xi + 0.5 * dt * k2, lag_mid)
        k4 = f(xi + dt * k3, lag_next)
        x[i + 1] = xi + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    series = x[hist_len + 1 :: stride]
    return series[discard : discard + n]


def memory_capacity_input(n: int, seed: int = 0) -> FloatArray:
    """I.i.d. uniform input in [-1, 1] for the linear memory-capacity measurement."""
    return np.random.default_rng(seed).uniform(-1.0, 1.0, n)
