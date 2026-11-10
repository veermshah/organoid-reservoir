import pytest

from organoid_reservoir.config import ExperimentConfig, get_dotted, load_config, set_dotted


def test_defaults_and_hash_stable():
    a, b = ExperimentConfig(), ExperimentConfig()
    assert a.hash() == b.hash() and len(a.hash()) == 12


def test_hash_changes_with_settings_but_not_paths():
    a, b = ExperimentConfig(), ExperimentConfig()
    b.results_dir = "elsewhere"
    assert a.hash() == b.hash()
    b.reservoir.n_neurons += 1
    assert a.hash() != b.hash()


def test_yaml_and_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("name: t\nreservoir:\n  n_neurons: 50\n  stdp:\n    a_plus: 0.5\n")
    cfg = load_config(path, ["esn.leak_rate=0.3", "seeds=[7]"])
    assert cfg.name == "t" and cfg.reservoir.n_neurons == 50
    assert cfg.reservoir.stdp.a_plus == 0.5
    assert cfg.reservoir.w_exc == ExperimentConfig().reservoir.w_exc
    assert cfg.esn.leak_rate == 0.3 and cfg.seeds == [7]


def test_unknown_keys_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("reservoir:\n  bogus: 1\n")
    with pytest.raises(KeyError):
        load_config(path)
    with pytest.raises(KeyError):
        set_dotted(ExperimentConfig(), "reservoir.bogus", 1)


def test_dotted_access():
    cfg = ExperimentConfig()
    set_dotted(cfg, "reservoir.conn_lambda", 0.7)
    assert get_dotted(cfg, "reservoir.conn_lambda") == 0.7


def test_shipped_configs_load():
    for name in ("default", "quick", "sweep"):
        load_config(f"configs/{name}.yaml")
