import numpy as np
import pytest

from organoid_reservoir.encoding import (
    ElectrodeEncoder,
    EncoderConfig,
    fibonacci_sphere,
    pad_sequences,
)


def test_intensity_shapes_and_range():
    enc = ElectrodeEncoder(12, EncoderConfig(electrodes_per_input=4))
    enc.fit(np.random.default_rng(0).standard_normal((100, 12)))
    u = np.random.default_rng(1).standard_normal((5, 9, 12)) * 3
    s = enc.intensities(u)
    assert enc.n_electrodes == 48
    assert s.shape == (5, 9, 48)
    assert s.min() >= 0.0 and s.max() < 1.0


def test_push_pull_polarity():
    enc = ElectrodeEncoder(1, EncoderConfig(electrodes_per_input=2))
    pos = enc.intensities(np.array([[2.0]]))[0]
    neg = enc.intensities(np.array([[-2.0]]))[0]
    assert pos[0] > 0 and pos[1] == 0
    assert neg[0] == 0 and neg[1] > 0


def test_staggered_thresholds_recruit_more_electrodes():
    enc = ElectrodeEncoder(1, EncoderConfig(electrodes_per_input=8, max_threshold=1.5))
    small = (enc.intensities(np.array([[0.2]])) > 0).sum()
    large = (enc.intensities(np.array([[2.0]])) > 0).sum()
    assert large > small >= 1


def test_odd_electrode_count_rejected():
    with pytest.raises(ValueError):
        ElectrodeEncoder(2, EncoderConfig(electrodes_per_input=3))


def test_fibonacci_sphere_on_surface():
    pts = fibonacci_sphere(24, radius=1.0)
    assert pts.shape == (24, 3)
    np.testing.assert_allclose(np.linalg.norm(pts, axis=1), 1.0)


def test_pad_sequences():
    seqs = [np.ones((3, 2)), np.ones((5, 2)) * 2]
    out, lengths = pad_sequences(seqs)
    assert out.shape == (2, 5, 2)
    np.testing.assert_array_equal(lengths, [3, 5])
    assert out[0, 3:].sum() == 0
