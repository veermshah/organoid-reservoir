"""Experiment configuration: nested dataclasses loaded from YAML.

Every results file is stamped with a short hash of the fully resolved config
and the current git commit, so a number in a table can be traced back to the
exact settings that produced it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

from .encoding import EncoderConfig
from .esn import ESNConfig
from .lstm import LSTMConfig
from .reservoir import ReservoirConfig

T = TypeVar("T")


@dataclass
class TimeSeriesConfig:
    """Lengths and options shared by the time-series tasks."""

    washout: int = 100
    n_train: int = 2000
    n_test: int = 1000
    electrodes_per_input: int = 8
    ar_lags: int = 10
    mackey_horizon: int = 1
    memory_max_delay: int = 40


@dataclass
class SizeSweepConfig:
    sizes: list[int] = field(default_factory=lambda: [100, 200, 400, 800])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])


@dataclass
class DiagnosticsConfig:
    gains: list[float] = field(default_factory=lambda: [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
    n_streams: int = 120
    stream_frames: int = 15
    noise_level: float = 0.3
    separation_distances: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 1.0])


@dataclass
class HyperSweepConfig:
    """Grid sweep over two dotted config keys, scored on one task."""

    task: str = "vowels"
    x_key: str = "reservoir.frac_exc"
    x_values: list[float] = field(default_factory=lambda: [0.6, 0.7, 0.8, 0.9])
    y_key: str = "reservoir.conn_lambda"
    y_values: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.5, 0.8])
    n_neurons: int = 200
    seeds: list[int] = field(default_factory=lambda: [0, 1])


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    name: str = "default"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    data_dir: str = "data/raw"
    results_dir: str = "results"
    figures_dir: str = "figures"
    stdp_epochs: int = 1
    stdp_batch_size: int = 30
    reservoir: ReservoirConfig = field(default_factory=ReservoirConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    esn: ESNConfig = field(default_factory=ESNConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    timeseries: TimeSeriesConfig = field(default_factory=TimeSeriesConfig)
    size_sweep: SizeSweepConfig = field(default_factory=SizeSweepConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    hyper_sweep: HyperSweepConfig = field(default_factory=HyperSweepConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def hash(self) -> str:
        """Short SHA-256 of the resolved config (paths excluded)."""
        d = self.to_dict()
        for k in ("results_dir", "figures_dir", "data_dir"):
            d.pop(k)
        blob = json.dumps(d, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def _from_dict(cls: type[T], data: dict[str, Any]) -> T:
    if not dataclasses.is_dataclass(cls):
        raise TypeError(cls)
    hints = {f.name: f for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in hints:
            raise KeyError(f"unknown config key {cls.__name__}.{key}")
        default = getattr(cls(), key)
        if dataclasses.is_dataclass(default) and isinstance(value, dict):
            kwargs[key] = _from_dict(type(default), value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(
    path: str | Path | None = None, overrides: list[str] | None = None
) -> ExperimentConfig:
    """Load a YAML config (missing keys keep their defaults) and apply ``key=value`` overrides."""
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text()) or {}
    cfg = _from_dict(ExperimentConfig, raw)
    for item in overrides or []:
        key, _, value = item.partition("=")
        set_dotted(cfg, key.strip(), yaml.safe_load(value))
    return cfg


def set_dotted(obj: Any, key: str, value: Any) -> None:
    """Set a nested attribute such as ``reservoir.n_neurons`` on a config object."""
    *parents, last = key.split(".")
    for p in parents:
        obj = getattr(obj, p)
    if not hasattr(obj, last):
        raise KeyError(f"unknown config key {key}")
    setattr(obj, last, value)


def get_dotted(obj: Any, key: str) -> Any:
    for p in key.split("."):
        obj = getattr(obj, p)
    return obj


def git_commit(cwd: str | Path | None = None) -> str:
    """Current commit hash with a ``-dirty`` suffix if tracked files changed, or ``unknown``."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short=10", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("-dirty" if dirty else "")
