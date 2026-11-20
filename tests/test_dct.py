import numpy as np

from videowm.dct import C, LOW_BAND, ZIGZAG, block_dct, block_idct, get_band, set_band

rng = np.random.default_rng(1)


def test_dct_matrix_is_orthonormal():
    assert np.allclose(C @ C.T, np.eye(8))


def test_zigzag_matches_jpeg_order():
    assert ZIGZAG[:10] == [(0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1), (3, 0)]
    assert ZIGZAG[-1] == (7, 7) and len(set(ZIGZAG)) == 64
    assert [ZIGZAG[i] for i in LOW_BAND] == [(0, 1), (1, 0), (2, 0), (1, 1), (0, 2)]


def test_dc_of_flat_block():
    d = block_dct(np.full((8, 8), 100, np.uint8))
    assert np.isclose(d[0, 0, 0, 0], 800.0) and np.allclose(d[0, 0].ravel()[1:], 0)


def test_roundtrip_and_border_untouched():
    y = rng.integers(0, 256, (50, 70), dtype=np.uint8)
    d = block_dct(y)
    assert d.shape == (6, 8, 8, 8)
    assert np.array_equal(block_idct(d, y), y)


def test_band_get_set():
    y = rng.integers(0, 256, (48, 64), dtype=np.uint8)
    d = block_dct(y)
    vals = get_band(d)
    assert vals.shape == (48, 5)
    d2 = set_band(d, vals + 1.0)
    assert np.allclose(get_band(d2) - vals, 1.0)
    other = np.ones((8, 8), bool)
    for r, c in [ZIGZAG[i] for i in LOW_BAND]:
        other[r, c] = False
    assert np.array_equal(d2[:, :, other], d[:, :, other])
