import numpy as np

from organoid_reservoir.reservoir import ReservoirConfig, SpikingReservoir, STDPConfig


def _stim(enc, B=4, T=6, seed=0):
    rng = np.random.default_rng(seed)
    return enc.intensities(rng.standard_normal((B, T, enc.n_inputs)) * 2)


def test_same_seed_same_spikes(small_encoder):
    cfg = ReservoirConfig(n_neurons=60, mean_in_degree=10)
    a = SpikingReservoir(cfg, small_encoder, seed=3)
    b = SpikingReservoir(cfg, small_encoder, seed=3)
    stim = _stim(small_encoder)
    ra = a.run(stim, record_spikes=True)
    rb = b.run(stim, record_spikes=True)
    np.testing.assert_array_equal(a.W, b.W)
    np.testing.assert_array_equal(ra["spikes"], rb["spikes"])
    np.testing.assert_array_equal(ra["states"], rb["states"])
    assert ra["spikes"].any()


def test_different_seed_differs(small_encoder):
    cfg = ReservoirConfig(n_neurons=60, mean_in_degree=10)
    a = SpikingReservoir(cfg, small_encoder, seed=1)
    b = SpikingReservoir(cfg, small_encoder, seed=2)
    assert not np.array_equal(a.W, b.W)


def test_output_shapes(small_reservoir, small_encoder):
    stim = _stim(small_encoder, B=3, T=5)
    out = small_reservoir.run(stim, record_spikes=True)
    n = small_reservoir.n_neurons
    assert out["states"].shape == (3, 5, n)
    assert out["counts"].shape == (3, 5, n)
    assert out["spikes"].shape == (3, 5 * small_encoder.config.frame_steps, n)


def test_multi_timescale_states(small_encoder):
    cfg = ReservoirConfig(n_neurons=40, mean_in_degree=8, trace_taus=[10.0, 50.0])
    res = SpikingReservoir(cfg, small_encoder, seed=0)
    out = res.run(_stim(small_encoder, B=2, T=3))
    assert out["states"].shape == (2, 3, 80)


def test_connectivity_is_distance_dependent_and_signed(small_encoder):
    res = SpikingReservoir(ReservoirConfig(n_neurons=200, mean_in_degree=20), small_encoder, 0)
    d = np.linalg.norm(res.positions[:, None] - res.positions[None], axis=-1)
    assert d[res.mask].mean() < d[~res.mask].mean()
    assert (res.W[res.is_exc] >= 0).all() and (res.W[~res.is_exc] <= 0).all()
    assert not np.diag(res.mask).any()
    assert abs(res.mask.sum(0).mean() - 20) < 4


def test_stdp_changes_only_existing_ee_synapses(small_encoder):
    cfg = ReservoirConfig(n_neurons=80, mean_in_degree=15, stdp=STDPConfig(a_plus=0.05))
    res = SpikingReservoir(cfg, small_encoder, seed=0)
    w0 = res.W.copy()
    res.run(_stim(small_encoder, B=2, T=20), plastic=True)
    ne = res.n_exc
    changed = w0 != res.W
    assert changed[:ne, :ne].any()
    assert not changed[ne:, :].any() and not changed[:, ne:].any()
    assert not changed[~res.mask].any()
    assert (res.W[:ne, :ne] >= 0).all()
    res.reset_weights()
    np.testing.assert_array_equal(res.W, w0)


def test_transform_matches_run_without_plasticity(small_reservoir, small_encoder):
    x = np.random.default_rng(5).standard_normal((2, 4, 3))
    small_reservoir.reseed_noise(11)
    a = small_reservoir.transform(x)
    small_reservoir.reseed_noise(11)
    b = small_reservoir.run(small_encoder.intensities(x))["states"]
    np.testing.assert_array_equal(a, b)
