from pathlib import Path

import numpy as np

from organoid_reservoir import data, diagnostics, experiments
from organoid_reservoir.config import ExperimentConfig

FIXTURES = Path(__file__).parent / "fixtures"


def tiny_cfg() -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.reservoir.n_neurons = 50
    cfg.reservoir.mean_in_degree = 10
    cfg.esn.n_neurons = 50
    cfg.encoder.frame_steps = 5
    cfg.lstm.enabled = False
    cfg.timeseries.washout = 20
    cfg.timeseries.n_train = 150
    cfg.timeseries.n_test = 80
    cfg.timeseries.memory_max_delay = 5
    return cfg


def test_vowels_pipeline_on_fixture():
    ds = data.load_vowel_files(
        FIXTURES / "ae_tiny.train", FIXTURES / "ae_tiny.test", (3, 3), (2, 2)
    )
    rows = experiments.vowels_seed(
        tiny_cfg(), ds, 0, models=("spiking", "spiking_stdp", "esn", "logreg_mean_std")
    )
    accs = {r["model"]: r["value"] for r in rows if r["metric"] == "accuracy"}
    assert set(accs) == {"spiking", "spiking_stdp", "esn", "logreg_mean_std"}
    assert all(0.0 <= v <= 1.0 for v in accs.values())
    assert any(r["metric"] == "weight_change" and r["value"] > 0 for r in rows)


def test_timeseries_pipeline_esn_beats_linear_on_henon():
    rows, preds = experiments.timeseries_seed(
        tiny_cfg(), "henon", 0, models=("spiking", "esn", "linear_ar"), keep_predictions=True
    )
    err = {r["model"]: r["value"] for r in rows}
    assert err["esn"] < err["linear_ar"]
    assert np.isfinite(err["spiking"])
    assert preds["esn"].shape == preds["target"].shape == (80,)


def test_series_tasks_have_matching_lengths():
    cfg = tiny_cfg()
    for name in (*experiments.TS_TASKS, "memory"):
        task = experiments.make_series_task(cfg, name, seed=1)
        assert len(task.u) == len(task.y) == 250


def test_memory_capacity_positive_for_esn():
    rows, curves = experiments.memory_seed(tiny_cfg(), 0)
    mc = {r["model"]: r["value"] for r in rows}
    assert mc["esn"] > 1.0
    assert curves["spiking"].shape == (5,)


def test_effective_rank():
    rng = np.random.default_rng(0)
    low = rng.standard_normal((40, 2)) @ rng.standard_normal((2, 30))
    assert diagnostics.effective_rank(low) == 2
    assert diagnostics.effective_rank(np.zeros((5, 5))) == 0


def test_separation_grows_with_input_distance():
    cfg = tiny_cfg()
    enc = experiments.build_encoder(cfg, 1, np.linspace(-1, 1, 50)[:, None])
    esn = experiments.build_esn(cfg, enc, 0)
    curves = diagnostics.separation_curves(esn, [0.0, 0.5, 1.0], n_frames=6, n_pairs=4)
    assert curves[0.0].max() == 0.0
    assert curves[1.0][0] > curves[0.5][0] > 0
