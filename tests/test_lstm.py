import numpy as np
import pytest

from organoid_reservoir.lstm import LSTMConfig, lstm_classify, lstm_forecast, torch_available

pytestmark = pytest.mark.skipif(not torch_available(), reason="torch not installed")


def test_lstm_classifies_toy_sequences():
    rng = np.random.default_rng(0)
    X = [rng.standard_normal((int(rng.integers(5, 9)), 2)) + (i % 2) * 2 for i in range(40)]
    y = np.array([i % 2 for i in range(40)])
    pred = lstm_classify(X[:30], y[:30], X[30:], LSTMConfig(hidden=8, epochs=30), seed=0)
    assert pred.shape == (10,)
    assert (pred == y[30:]).mean() >= 0.9


def test_lstm_forecast_shape():
    u = np.sin(np.arange(300) * 0.2)
    y = np.roll(u, -1)
    p = lstm_forecast(u, y, 20, 200, LSTMConfig(hidden=8, epochs=5, window=5), seed=0)
    assert p.shape == (80,)
