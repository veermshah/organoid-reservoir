from pathlib import Path

import numpy as np

from organoid_reservoir import data

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_fixture_blocks():
    ds = data.load_vowel_files(
        FIXTURES / "ae_tiny.train", FIXTURES / "ae_tiny.test", (3, 3), (2, 2)
    )
    assert len(ds.X_train) == 6 and len(ds.X_test) == 4
    assert ds.n_features == data.N_COEFFS
    assert ds.n_classes == 2
    assert all(x.ndim == 2 and x.shape[1] == 12 and 7 <= len(x) <= 29 for x in ds.X_train)
    np.testing.assert_array_equal(ds.y_train, [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(ds.y_test, [0, 0, 1, 1])


def test_block_count_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        data.load_vowel_files(FIXTURES / "ae_tiny.train", FIXTURES / "ae_tiny.test", (3, 4), (2, 2))


def test_parse_handles_whitespace_only_separators():
    text = "1 2 3\n4 5 6\n   \n7 8 9\n\n\n"
    blocks = data.parse_vowel_blocks(text)
    assert [b.shape for b in blocks] == [(2, 3), (1, 3)]


def test_label_sizes_match_uci_split():
    assert len(data.labels_from_sizes(data.TRAIN_SIZES)) == 270
    assert len(data.labels_from_sizes(data.TEST_SIZES)) == 370


def test_henon_is_deterministic_and_bounded():
    x1 = data.henon_map(500)
    x2 = data.henon_map(500)
    np.testing.assert_array_equal(x1, x2)
    assert np.all(np.abs(x1) < 1.5)
    # the map relation holds: x[t+1] = 1 - 1.4 x[t]^2 + 0.3 x[t-1]
    np.testing.assert_allclose(x1[2:], 1 - 1.4 * x1[1:-1] ** 2 + 0.3 * x1[:-2], atol=1e-12)


def test_narma10_shapes_and_range():
    u, y = data.narma10(1000, seed=1)
    assert u.shape == y.shape == (1000,)
    assert 0 <= u.min() and u.max() <= 0.5
    assert np.isfinite(y).all() and 0 < y.mean() < 1


def test_mackey_glass_is_chaotic_but_bounded():
    x = data.mackey_glass(400)
    assert x.shape == (400,)
    assert 0.2 < x.min() and x.max() < 1.6
    assert x.std() > 0.1


def test_memory_capacity_input():
    u = data.memory_capacity_input(100, seed=3)
    assert u.shape == (100,) and np.abs(u).max() <= 1
