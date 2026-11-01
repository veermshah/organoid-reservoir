import numpy as np

from organoid_reservoir.encoding import ElectrodeEncoder
from organoid_reservoir.esn import EchoStateNetwork, ESNConfig
from organoid_reservoir.features import mean_std_features, pool_sequence_states, resample_sequences
from organoid_reservoir.readout import (
    accuracy,
    nrmse,
    train_logistic_readout,
    train_ridge_readout,
)


def test_logistic_readout_learns_toy_task():
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((3, 10)) * 3
    y = np.repeat(np.arange(3), 40)
    X = centers[y] + rng.standard_normal((120, 10))
    model = train_logistic_readout(X[::2], y[::2])
    assert accuracy(y[1::2], model.predict(X[1::2])) > 0.95


def test_ridge_readout_recovers_linear_map():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 5))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0, 3.0]) + 0.01 * rng.standard_normal(300)
    model = train_ridge_readout(X[:200], y[:200])
    assert nrmse(y[200:], model.predict(X[200:])) < 0.05


def test_spiking_states_feed_readout(small_reservoir):
    # Two input classes with opposite sign should be separable from reservoir states.
    rng = np.random.default_rng(2)
    y = np.repeat([0, 1], 12)
    x = np.where(y[:, None, None] == 0, 1.5, -1.5) + 0.3 * rng.standard_normal((24, 4, 3))
    states = small_reservoir.transform(x)
    feats = pool_sequence_states(states, np.full(24, 4))
    model = train_logistic_readout(feats[::2], y[::2])
    assert accuracy(y[1::2], model.predict(feats[1::2])) >= 0.9


def test_nrmse_and_accuracy_values():
    assert nrmse(np.array([1.0, -1.0]), np.array([1.0, -1.0])) == 0.0
    assert accuracy(np.array([0, 1, 1]), np.array([0, 1, 0])) == 2 / 3


def test_pooling_handles_variable_lengths():
    states = np.arange(2 * 6 * 2, dtype=float).reshape(2, 6, 2)
    feats = pool_sequence_states(states, np.array([3, 6]), n_segments=3)
    assert feats.shape == (2, 8)
    np.testing.assert_allclose(feats[0, :2], states[0, :3].mean(0))


def test_baseline_feature_shapes():
    seqs = [np.ones((7, 12)), np.ones((20, 12))]
    assert mean_std_features(seqs).shape == (2, 24)
    assert resample_sequences(seqs, 8).shape == (2, 96)


def test_esn_shapes_and_determinism():
    enc = ElectrodeEncoder(2).fit(np.random.default_rng(0).standard_normal((30, 2)))
    a = EchoStateNetwork(ESNConfig(n_neurons=50), enc, seed=4)
    b = EchoStateNetwork(ESNConfig(n_neurons=50), enc, seed=4)
    x = np.random.default_rng(1).standard_normal((3, 7, 2))
    sa, sb = a.transform(x), b.transform(x)
    assert sa.shape == (3, 7, 50)
    np.testing.assert_array_equal(sa, sb)
    assert np.abs(np.linalg.eigvals(a.W.astype(float))).max() < 0.91
