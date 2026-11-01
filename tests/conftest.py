import numpy as np
import pytest

from organoid_reservoir.encoding import ElectrodeEncoder, EncoderConfig
from organoid_reservoir.reservoir import ReservoirConfig, SpikingReservoir


@pytest.fixture
def small_encoder() -> ElectrodeEncoder:
    rng = np.random.default_rng(0)
    return ElectrodeEncoder(3, EncoderConfig(frame_steps=5)).fit(rng.standard_normal((50, 3)))


@pytest.fixture
def small_reservoir(small_encoder: ElectrodeEncoder) -> SpikingReservoir:
    return SpikingReservoir(ReservoirConfig(n_neurons=60, mean_in_degree=10), small_encoder, seed=7)
