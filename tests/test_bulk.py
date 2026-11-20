import os

import pytest

from videowm import bulk


def test_present_ctr_roundtrip_and_fresh_nonce():
    k, d = os.urandom(16), os.urandom(5000)
    a, b = bulk.present_ctr_encrypt(k, d), bulk.present_ctr_encrypt(k, d)
    assert a != b and len(a) == len(d) + 8
    assert bulk.present_ctr_decrypt(k, a) == d == bulk.present_ctr_decrypt(k, b)


def test_ff1_chunks_roundtrip_and_position_tweak():
    k = os.urandom(16)
    d = bytes(8) * 4
    c = bulk.ff1_encrypt(k, d)
    assert len(c) == len(d) and len({c[i:i + 8] for i in range(0, 32, 8)}) == 4
    assert bulk.ff1_decrypt(k, c) == d
    with pytest.raises(ValueError):
        bulk.ff1_encrypt(k, b"123")
