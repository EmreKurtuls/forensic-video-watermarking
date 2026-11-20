import os

import numpy as np
import pytest

from videowm.keys import derive_film_keys, generate_master_key
from videowm.payload import (Algorithm, NoValidMark, PayloadCodec, pack_block, unpack_block)

FILM = b"BBB-0001"
ALGS = [Algorithm.PRESENT128, Algorithm.FF1]
rng = np.random.default_rng(int.from_bytes(os.urandom(8), "big"))


@pytest.fixture(scope="module")
def keys():
    return derive_film_keys(generate_master_key(), FILM)


def flip(bits, n):
    b = bits.copy()
    idx = rng.choice(len(b), n, replace=False)
    b[idx] ^= 1
    return b


def test_block_layout():
    blk = pack_block(0xDEADBEEF, 0xABCDEF)
    assert blk == 0x1DEADBEEFABCDEF0
    assert unpack_block(blk) == (1, 0xDEADBEEF, 0xABCDEF, 0)
    for bad in [(2 ** 32, 0), (0, 2 ** 24), (-1, 0)]:
        with pytest.raises(ValueError):
            pack_block(*bad)


@pytest.mark.parametrize("alg", ALGS)
def test_roundtrip_and_size(keys, alg):
    codec = PayloadCodec(keys, alg)
    for uid, sess in [(0, 0), (1001, 7), (2 ** 32 - 1, 2 ** 24 - 1)]:
        bits = codec.encode(uid, sess)
        assert bits.shape == (188,) and set(np.unique(bits)) <= {0, 1}
        ident = codec.decode(bits)
        assert (ident.user_id, ident.session, ident.corrected_bits) == (uid, sess, 0)


@pytest.mark.parametrize("alg", ALGS)
def test_corrects_up_to_12_bit_errors(keys, alg):
    codec = PayloadCodec(keys, alg)
    bits = codec.encode(1001, 7)
    for n in range(1, 13):
        ident = codec.decode(flip(bits, n))
        assert (ident.user_id, ident.session, ident.corrected_bits) == (1001, 7, n)


@pytest.mark.parametrize("alg", ALGS)
def test_too_many_errors_gives_no_valid_mark(keys, alg):
    codec = PayloadCodec(keys, alg)
    bits = codec.encode(1001, 7)
    for _ in range(50):
        with pytest.raises(NoValidMark):
            codec.decode(flip(bits, 40))


@pytest.mark.parametrize("alg", ALGS)
def test_wrong_key_never_yields_an_identity(keys, alg):
    bits = PayloadCodec(keys, alg).encode(1001, 7)
    for _ in range(20):
        with pytest.raises(NoValidMark):
            PayloadCodec(derive_film_keys(generate_master_key(), FILM), alg).decode(bits)
    with pytest.raises(NoValidMark):
        PayloadCodec(derive_film_keys(generate_master_key(), b"SIN-0001"), alg).decode(bits)


def test_algorithm_is_bound_by_mac(keys):
    bits = PayloadCodec(keys, Algorithm.PRESENT128).encode(1001, 7)
    with pytest.raises(NoValidMark):
        PayloadCodec(keys, Algorithm.FF1).decode(bits)


@pytest.mark.parametrize("alg", ALGS)
def test_decrypt_not_called_when_mac_fails(keys, alg):
    codec = PayloadCodec(keys, alg)
    bits = PayloadCodec(derive_film_keys(generate_master_key(), FILM), alg).encode(1001, 7)

    def boom(_):
        raise AssertionError("decrypt called before MAC check")
    codec._dec = boom
    with pytest.raises(NoValidMark):
        codec.decode(bits)


@pytest.mark.parametrize("alg", ALGS)
def test_random_bits_rejected(keys, alg):
    codec = PayloadCodec(keys, alg)
    for _ in range(500):
        with pytest.raises(NoValidMark):
            codec.decode(rng.integers(0, 2, 188, dtype=np.uint8))


def test_wrong_length_rejected(keys):
    codec = PayloadCodec(keys, Algorithm.PRESENT128)
    with pytest.raises(ValueError):
        codec.decode(np.zeros(192, np.uint8))


def test_sessions_give_unrelated_codewords(keys):
    codec = PayloadCodec(keys, Algorithm.PRESENT128)
    a, b = codec.encode(1001, 7), codec.encode(1001, 8)
    assert 60 < int(np.sum(a != b)) < 130


def test_uniform_error_message(keys):
    codec = PayloadCodec(keys, Algorithm.PRESENT128)
    msgs = set()
    for bits in [flip(codec.encode(1, 1), 40), PayloadCodec(derive_film_keys(generate_master_key(), FILM),
                                                              Algorithm.PRESENT128).encode(1, 1)]:
        with pytest.raises(NoValidMark) as e:
            codec.decode(bits)
        msgs.add(str(e.value))
    assert msgs == {"no valid watermark found"}


@pytest.mark.parametrize("alg", ALGS)
def test_wrong_enc_key_alone_is_rejected_before_decrypt(keys, alg):
    from videowm.keys import FilmKeys
    bits = PayloadCodec(keys, alg).encode(1001, 7)
    for _ in range(2000 if alg is Algorithm.PRESENT128 else 300):
        bad = FilmKeys(keys.film_id, os.urandom(16), keys.mac_key, keys.stego_key)
        codec = PayloadCodec(bad, alg)
        codec._dec = lambda _: (_ for _ in ()).throw(AssertionError("decrypt reached"))
        with pytest.raises(NoValidMark):
            codec.decode(bits)
