"""Command-line entry point.

Examples
--------
    python -m organoid_reservoir.run --task all
    python -m organoid_reservoir.run --task vowels --config configs/quick.yaml
    python -m organoid_reservoir.run --task henon --set reservoir.n_neurons=200 --seeds 0 1
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import data, diagnostics, experiments, plotting, viz3d
from .config import ExperimentConfig, git_commit, load_config
from .encoding import pad_sequences

TASKS = (
    "vowels",
    "henon",
    "narma10",
    "mackey_glass",
    "timeseries",
    "memory",
    "sizes",
    "diagnostics",
    "all",
)


class Runner:
    """Runs tasks for one resolved config and writes stamped CSVs and figures."""

    def __init__(self, cfg: ExperimentConfig, verbose: bool = True) -> None:
        self.cfg = cfg
        self.results = Path(cfg.results_dir)
        self.figures = Path(cfg.figures_dir)
        self.results.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)
        self.stamp = {"config_hash": cfg.hash(), "git_commit": git_commit()}
        self.verbose = verbose
        self._vowels: data.SequenceDataset | None = None

    # -- helpers ---------------------------------------------------------
    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def vowels(self) -> data.SequenceDataset:
        if self._vowels is None:
            self._vowels = data.load_japanese_vowels(self.cfg.data_dir)
        return self._vowels

    def save(
        self, rows: list[dict[str, Any]], name: str, replace: Iterable[str] = ()
    ) -> pd.DataFrame:
        """Write rows to ``results/<name>.csv``, replacing earlier rows of the same tasks."""
        path = self.results / f"{name}.csv"
        df = pd.DataFrame(rows)
        for k, v in self.stamp.items():
            df[k] = v
        replace = set(replace)
        if path.exists() and replace:
            old = pd.read_csv(path)
            old = old[~old.task.isin(replace)]
            df = pd.concat([old, df], ignore_index=True)
        df.to_csv(path, index=False, float_format="%.6g")
        return df

    def write_meta(self, task: str, seconds: float) -> None:
        path = self.results / "run_meta.json"
        meta: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
        meta[task] = {
            **self.stamp,
            "config_name": self.cfg.name,
            "seeds": self.cfg.seeds,
            "seconds": round(seconds, 1),
            "finished": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        path.write_text(json.dumps(meta, indent=2) + "\n")
        (self.results / f"config_{self.stamp['config_hash']}.json").write_text(
            json.dumps(self.cfg.to_dict(), indent=2) + "\n"
        )

    # -- tasks -----------------------------------------------------------
    def run_vowels(self) -> None:
        ds = self.vowels()
        rows = []
        for seed in self.cfg.seeds:
            t0 = time.perf_counter()
            r = experiments.vowels_seed(self.cfg, ds, seed)
            rows += r
            accs = ", ".join(
                f"{x['model']}={x['value']:.3f}" for x in r if x["metric"] == "accuracy"
            )
            self.log(f"  vowels seed {seed} ({time.perf_counter() - t0:.0f}s): {accs}")
        df = self.save(rows, "vowels_runs", replace=["vowels"])
        plotting.plot_vowels(df, self.figures / "vowels_accuracy.png")
        self.vowel_figures(ds)

    def vowel_figures(self, ds: data.SequenceDataset) -> None:
        """Raster plot, 3D snapshot and activity GIF for one utterance."""
        cfg = self.cfg
        frames = np.concatenate(ds.X_train)
        enc = experiments.build_encoder(cfg, ds.n_features, frames)
        res = experiments.build_spiking(cfg, enc, seed=cfg.seeds[0])
        X, _ = pad_sequences([ds.X_test[0]])
        stim = enc.intensities(X)
        out = res.run(stim, record_spikes=True)
        spikes = np.asarray(out["spikes"])[0]
        plotting.plot_raster(
            spikes,
            stim[0],
            res.is_exc,
            enc.config.frame_steps,
            self.figures / "raster_vowel.png",
            "One utterance (speaker 1) delivered as electrode pulse trains",
        )
        Xall, lall = pad_sequences(ds.X_test[::10])
        counts = np.asarray(res.run(enc.intensities(Xall), lall)["counts"])
        secs = float(lall.sum()) * enc.config.frame_steps / 1000.0
        rate = np.asarray(counts.sum((0, 1)), dtype=np.float64) / secs
        viz3d.plot_reservoir_3d(res, rate, self.figures / "reservoir_3d.png")
        viz3d.animate_reservoir(
            res, spikes, stim[0], enc.config.frame_steps, self.figures / "reservoir_activity.gif"
        )

    def run_timeseries(self, tasks: Iterable[str]) -> None:
        tasks = list(tasks)
        rows = []
        for task in tasks:
            for seed in self.cfg.seeds:
                t0 = time.perf_counter()
                keep = task == "henon" and seed == self.cfg.seeds[0]
                r, preds = experiments.timeseries_seed(self.cfg, task, seed, keep_predictions=keep)
                rows += r
                errs = ", ".join(
                    f"{x['model']}={x['value']:.3f}" for x in r if x["metric"] == "nrmse"
                )
                self.log(f"  {task} seed {seed} ({time.perf_counter() - t0:.0f}s): {errs}")
                if keep:
                    pd.DataFrame(preds).iloc[:200].to_csv(
                        self.results / "henon_predictions.csv", index=False, float_format="%.6g"
                    )
                    plotting.plot_predictions(preds, self.figures / "henon_prediction.png")
        df = self.save(rows, "timeseries_runs", replace=tasks)
        plotting.plot_timeseries(df, self.figures / "timeseries_nrmse.png")

    def run_memory(self) -> None:
        rows, curve_rows = [], []
        for seed in self.cfg.seeds:
            r, curves = experiments.memory_seed(self.cfg, seed)
            rows += r
            for model, c in curves.items():
                curve_rows += [
                    {"task": "memory", "model": model, "seed": seed, "delay": k + 1, "r2": v}
                    for k, v in enumerate(c)
                ]
            self.log(
                f"  memory seed {seed}: " + ", ".join(f"{x['model']}={x['value']:.2f}" for x in r)
            )
        self.save(rows, "memory_runs", replace=["memory"])
        cdf = self.save(curve_rows, "memory_curves", replace=["memory"])
        plotting.plot_memory(cdf, self.figures / "memory_capacity.png")

    def run_sizes(self) -> None:
        sc = self.cfg.size_sweep
        ds = self.vowels()
        rows = []
        for n in sc.sizes:
            for seed in sc.seeds:
                t0 = time.perf_counter()
                rows += experiments.vowels_seed(
                    self.cfg, ds, seed, models=("spiking", "spiking_dc", "esn"), n_neurons=n
                )
                r, _ = experiments.timeseries_seed(
                    self.cfg, "henon", seed, models=("spiking", "spiking_dc", "esn"), n_neurons=n
                )
                rows += r
                self.log(f"  size {n} seed {seed} ({time.perf_counter() - t0:.0f}s)")
        df = self.save(rows, "size_sweep")
        plotting.plot_size_sweep(df, self.figures / "size_sweep.png")

    def run_diagnostics(self) -> None:
        cfg = self.cfg
        dc = cfg.diagnostics
        rng = np.random.default_rng(0)
        fit = rng.uniform(-1, 1, (1000, 1))
        k = cfg.timeseries.electrodes_per_input
        enc_s = experiments.build_encoder(cfg, 1, fit, electrodes_per_input=k)
        enc_dc = experiments.build_encoder(cfg, 1, fit, electrodes_per_input=k, mode="current")
        enc_e = experiments.build_encoder(cfg, 1, fit)
        seeds = cfg.seeds[:3]

        # Firing statistics, spontaneous and driven.
        stat_rows = []
        for seed in seeds:
            res = experiments.build_spiking(cfg, enc_s, seed)
            silent = np.zeros((4, 100, enc_s.n_electrodes))
            driven = enc_s.intensities(np.random.default_rng(seed).uniform(-1, 1, (4, 100, 1)))
            for cond, stim in (("spontaneous", silent), ("driven", driven)):
                stats = diagnostics.firing_statistics(res, stim)
                stat_rows.append({"task": "firing", "condition": cond, "seed": seed, **stats})
        self.save(stat_rows, "diagnostics_firing", replace=["firing"])

        # Kernel quality and generalisation rank vs weight gain.
        kq_rows = []
        for seed in seeds:
            for gain in dc.gains:
                sres, esn = diagnostics.scaled_models(cfg, gain, enc_s, enc_e, seed)
                for name, model in (("spiking", sres), ("esn", esn)):
                    kq, gr = diagnostics.kernel_and_generalization(
                        model, dc.n_streams, dc.stream_frames, dc.noise_level, seed
                    )
                    kq_rows.append(
                        {
                            "task": "kernel",
                            "model": name,
                            "seed": seed,
                            "gain": gain,
                            "kernel_quality": kq,
                            "generalization_rank": gr,
                        }
                    )
        kdf = self.save(kq_rows, "diagnostics_kernel", replace=["kernel"])
        plotting.plot_kernel_quality(kdf, self.figures / "kernel_quality.png")

        # Separation property.
        sres = experiments.build_spiking(cfg, enc_s, seeds[0])
        esn = experiments.build_esn(cfg, enc_e, seeds[0])
        curves = {
            "spiking": diagnostics.separation_curves(sres, dc.separation_distances),
            "esn": diagnostics.separation_curves(esn, dc.separation_distances),
        }
        sep_rows = [
            {"task": "separation", "model": m, "distance": d, "frame": t + 1, "value": v}
            for m, cv in curves.items()
            for d, y in cv.items()
            for t, v in enumerate(y)
        ]
        self.save(sep_rows, "diagnostics_separation", replace=["separation"])
        plotting.plot_separation(curves, self.figures / "separation.png")

        # How much of the gap is internal noise? DC reservoir with scaled background noise.
        noise_rows = []
        for seed in seeds:
            for scale in (0.1, 0.25, 0.5, 1.0, 2.0):
                ncfg = copy.deepcopy(cfg)
                ncfg.reservoir.noise_exc *= scale
                ncfg.reservoir.noise_inh *= scale
                henon = experiments.make_series_task(ncfg, "henon", seed)
                mem = experiments.make_series_task(ncfg, "memory", seed)
                res = experiments.build_spiking(ncfg, enc_dc, seed)
                err, _ = experiments.fit_series_readout(
                    henon, experiments.series_states(res, henon.u)
                )
                mcurve = experiments.memory_curve(
                    mem, experiments.series_states(res, mem.u), cfg.timeseries.memory_max_delay
                )
                noise_rows.append(
                    {
                        "task": "noise",
                        "seed": seed,
                        "noise_scale": scale,
                        "henon_nrmse": err,
                        "memory_capacity": mcurve.sum(),
                    }
                )
            self.log(f"  noise ablation seed {seed} done")
        ndf = self.save(noise_rows, "diagnostics_noise", replace=["noise"])
        plotting.plot_noise_ablation(ndf, self.figures / "noise_ablation.png")

    # -- summary ---------------------------------------------------------
    def summarize(self) -> None:
        frames = []
        for name in ("vowels_runs", "timeseries_runs", "memory_runs"):
            path = self.results / f"{name}.csv"
            if path.exists():
                frames.append(pd.read_csv(path))
        if not frames:
            return
        df = pd.concat(frames, ignore_index=True)
        summary = (
            df.groupby(["task", "metric", "model"])
            .agg(
                mean=("value", "mean"),
                std=("value", "std"),
                n_seeds=("seed", "nunique"),
                n_neurons=("n_neurons", "first"),
                seconds=("seconds", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(self.results / "summary.csv", index=False, float_format="%.6g")
        if {"spiking", "spiking_stdp"} <= set(df.model):
            plotting.plot_stdp(df, self.figures / "stdp_ablation.png")
        lines = ["| task | metric | model | mean ± std | seeds |", "|---|---|---|---|---|"]
        for _, r in summary[summary.metric != "weight_change"].iterrows():
            label = plotting.MODEL_LABELS.get(r.model, r.model)
            lines.append(
                f"| {r.task} | {r.metric} | {label} | {r['mean']:.4f} ± "
                f"{0 if np.isnan(r['std']) else r['std']:.4f} | {r.n_seeds} |"
            )
        (self.results / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", choices=TASKS, default="all")
    parser.add_argument("--config", default=None, help="YAML config (defaults built in)")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="override config values, e.g. reservoir.n_neurons=200",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.set)
    if args.seeds:
        cfg.seeds = args.seeds
    runner = Runner(cfg, verbose=not args.quiet)
    runner.log(
        f"config '{cfg.name}' hash={runner.stamp['config_hash']} "
        f"commit={runner.stamp['git_commit']}"
    )

    plan: list[tuple[str, Any]] = []
    t = args.task
    if t in ("vowels", "all"):
        plan.append(("vowels", runner.run_vowels))
    if t in ("henon", "narma10", "mackey_glass"):
        plan.append((t, lambda: runner.run_timeseries([t])))
    if t in ("timeseries", "all"):
        plan.append(("timeseries", lambda: runner.run_timeseries(experiments.TS_TASKS)))
    if t in ("memory", "all"):
        plan.append(("memory", runner.run_memory))
    if t in ("sizes", "all"):
        plan.append(("sizes", runner.run_sizes))
    if t in ("diagnostics", "all"):
        plan.append(("diagnostics", runner.run_diagnostics))

    total = time.perf_counter()
    for name, fn in plan:
        t0 = time.perf_counter()
        runner.log(f"[{name}]")
        fn()
        runner.write_meta(name, time.perf_counter() - t0)
        runner.log(f"[{name}] done in {time.perf_counter() - t0:.0f}s")
    runner.summarize()
    runner.log(f"finished in {time.perf_counter() - total:.0f}s; results in {cfg.results_dir}/")


if __name__ == "__main__":
    main()
