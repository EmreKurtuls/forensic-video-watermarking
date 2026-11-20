import numpy as np
import pytest

from videowm.keys import _hkdf, derive_film_keys, generate_master_key, pattern


def test_hkdf_rfc5869_case3():
    okm = _hkdf(bytes([0x0B] * 22), b"", 42)
    assert okm.hex() == ("8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f"
                         "3c738d2d9d201395faa4b61a96c8")


def test_master_key_is_random_32_bytes():
    a, b = generate_master_key(), generate_master_key()
    assert len(a) == 32 and a != b


def test_film_keys_deterministic_separated_and_sized():
    m = generate_master_key()
    k1, k1b, k2 = derive_film_keys(m, b"BBB-0001"), derive_film_keys(m, b"BBB-0001"), derive_film_keys(m, b"SIN-0001")
    assert k1 == k1b
    assert (len(k1.enc_key), len(k1.mac_key), len(k1.stego_key)) == (16, 16, 32)
    assert len({k1.enc_key, k1.mac_key, k1.stego_key[:16]}) == 3
    assert k1.enc_key != k2.enc_key and k1.mac_key != k2.mac_key and k1.stego_key != k2.stego_key
    assert k1.enc_key.hex() not in repr(k1)


def test_bad_inputs_rejected():
    with pytest.raises(ValueError):
        derive_film_keys(bytes(16), b"x")
    with pytest.raises(ValueError):
        derive_film_keys(bytes(32), b"")


def test_pattern_properties():
    sk = derive_film_keys(generate_master_key(), b"BBB-0001").stego_key
    p = pattern(sk, 0, 1584, 5)
    assert p.shape == (1584, 5) and set(np.unique(p)) == {-1.0, 1.0}
    assert np.array_equal(p, pattern(sk, 0, 1584, 5))
    assert abs(p.mean()) < 0.05
    q = pattern(sk, 1, 1584, 5)
    assert abs((p * q).mean()) < 0.05
    other = pattern(bytes(32), 0, 1584, 5)
    assert abs((p * other).mean()) < 0.05
