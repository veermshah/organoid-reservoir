"""Grid sweep over two reservoir hyperparameters, scored on the vowels task.

python -m organoid_reservoir.sweep --config configs/sweep.yaml
"""

from __future__ import annotations

import argparse
import copy
import time
from typing import Any

import numpy as np
import pandas as pd

from . import data, experiments, plotting
from .config import ExperimentConfig, get_dotted, git_commit, load_config, set_dotted


def run_sweep(cfg: ExperimentConfig, verbose: bool = True) -> pd.DataFrame:
    """Evaluate the pulse-stimulated spiking reservoir on every grid point."""
    hs = cfg.hyper_sweep
    if hs.task != "vowels":
        raise ValueError("only the vowels task is supported for sweeps")
    ds = data.load_japanese_vowels(cfg.data_dir)
    rows = []
    for yv in hs.y_values:
        for xv in hs.x_values:
            point = copy.deepcopy(cfg)
            set_dotted(point, hs.x_key, xv)
            set_dotted(point, hs.y_key, yv)
            t0 = time.perf_counter()
            accs = []
            for seed in hs.seeds:
                r = experiments.vowels_seed(
                    point, ds, seed, models=("spiking",), n_neurons=hs.n_neurons
                )
                accs.append(r[0]["value"])
                rows.append(
                    {
                        "x_key": hs.x_key,
                        "x": xv,
                        "y_key": hs.y_key,
                        "y": yv,
                        "seed": seed,
                        "accuracy": r[0]["value"],
                    }
                )
            if verbose:
                print(
                    f"  {hs.x_key}={xv} {hs.y_key}={yv}: {np.mean(accs):.3f} "
                    f"({time.perf_counter() - t0:.0f}s)",
                    flush=True,
                )
    df = pd.DataFrame(rows)
    df["config_hash"] = cfg.hash()
    df["git_commit"] = git_commit()
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sweep.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set)
    hs = cfg.hyper_sweep
    print(
        f"sweep {hs.x_key} x {hs.y_key} on {hs.task}, n_neurons={hs.n_neurons}, "
        f"default {hs.x_key}={get_dotted(cfg, hs.x_key)}, config hash {cfg.hash()}"
    )
    df = run_sweep(cfg)
    out = f"{cfg.results_dir}/hyper_sweep.csv"
    df.to_csv(out, index=False, float_format="%.6g")
    grid = (
        df.groupby(["y", "x"])
        .accuracy.mean()
        .unstack("x")
        .reindex(index=hs.y_values, columns=hs.x_values)
        .to_numpy()
    )
    plotting.plot_hyper_sweep(
        grid,
        hs.x_values,
        hs.y_values,
        "fraction excitatory" if hs.x_key.endswith("frac_exc") else hs.x_key,
        "connection length constant" if hs.y_key.endswith("conn_lambda") else hs.y_key,
        f"Vowels accuracy, {hs.n_neurons}-neuron spiking reservoir",
        f"{cfg.figures_dir}/hyper_sweep.png",
    )
    best: Any = df.groupby(["x", "y"]).accuracy.mean().idxmax()
    print(f"wrote {out}; best {hs.x_key}={best[0]}, {hs.y_key}={best[1]}")


if __name__ == "__main__":
    main()
