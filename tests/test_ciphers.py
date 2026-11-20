import os

import pytest
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from videowm.ciphers import AES128, FF1, PRESENT128, siphash24, siphash24_bytes

h = bytes.fromhex


@pytest.mark.parametrize("key,pt,ct", [
    ("2b7e151628aed2a6abf7158809cf4f3c", "3243f6a8885a308d313198a2e0370734", "3925841d02dc09fbdc118597196a0b32"),
    ("000102030405060708090a0b0c0d0e0f", "00112233445566778899aabbccddeeff", "69c4e0d86a7b0430d8cdb78070b4c55a"),
    ("2b7e151628aed2a6abf7158809cf4f3c", "6bc1bee22e409f96e93d7e117393172a", "3ad77bb40d7a3660a89ecaf32466ef97"),
    ("2b7e151628aed2a6abf7158809cf4f3c", "ae2d8a571e03ac9c9eb76fac45af8e51", "f5d3d58503b9699de785895a96fdbaaf"),
    ("2b7e151628aed2a6abf7158809cf4f3c", "30c81c46a35ce411e5fbc1191a0a52ef", "43b1cd7f598ece23881b00e3ed030688"),
    ("2b7e151628aed2a6abf7158809cf4f3c", "f69f2445df4f9b17ad2b417be66c3710", "7b0c785e27e8ad3f8223207104725dd4"),
])
def test_aes_kat(key, pt, ct):
    assert AES128(h(key)).encrypt_block(h(pt)) == h(ct)


def test_aes_matches_reference_library_on_random_inputs():
    for _ in range(200):
        key, pt = os.urandom(16), os.urandom(16)
        ref = Cipher(algorithms.AES(key), modes.ECB()).encryptor().update(pt)
        assert AES128(key).encrypt_block(pt) == ref


@pytest.mark.parametrize("key,pt,ct", [
    ("00000000000000000000000000000000", "0000000000000000", "96db702a2e6900af"),
    ("00000000000000000000000000000000", "ffffffffffffffff", "3c6019e5e5edd563"),
    ("ffffffffffffffffffffffffffffffff", "0000000000000000", "13238c710272a5d8"),
    ("ffffffffffffffffffffffffffffffff", "ffffffffffffffff", "628d9fbd4218e5b4"),
])
def test_present_kat(key, pt, ct):
    c = PRESENT128(h(key))
    assert c.encrypt_block(h(pt)) == h(ct)
    assert c.decrypt_block(h(ct)) == h(pt)


def test_present_roundtrip_random():
    for _ in range(200):
        c = PRESENT128(os.urandom(16))
        x = int.from_bytes(os.urandom(8), "big")
        assert c.decrypt_int(c.encrypt_int(x)) == x


def test_present_ctr_roundtrip():
    c = PRESENT128(os.urandom(16))
    data = os.urandom(1000)
    enc = c.ctr(data, nonce=42)
    assert enc != data and len(enc) == len(data)
    assert c.ctr(enc, nonce=42) == data


FF1_KEY = "2b7e151628aed2a6abf7158809cf4f3c"


@pytest.mark.parametrize("radix,tweak,pt,ct", [
    (10, "", "0123456789", "2433477484"),
    (10, "39383736353433323130", "0123456789", "6124200773"),
    (36, "3737373770717273373737", "0123456789abcdefghi", "a9tv40mll9kdu509eum"),
])
def test_ff1_kat(radix, tweak, pt, ct):
    f = FF1(h(FF1_KEY), radix)
    assert f.encrypt_str(pt, h(tweak)) == ct
    assert f.decrypt_str(ct, h(tweak)) == pt


def test_ff1_64bit_roundtrip_and_format():
    f = FF1(os.urandom(16), radix=2)
    for _ in range(50):
        x = int.from_bytes(os.urandom(8), "big")
        c = f.encrypt_int(x, 64, b"BBB-0001")
        assert 0 <= c < 2 ** 64
        assert f.decrypt_int(c, 64, b"BBB-0001") == x


def test_ff1_tweak_changes_ciphertext():
    f = FF1(os.urandom(16), radix=2)
    x = 0x1000003E90000070
    assert f.encrypt_int(x, 64, b"film-A") != f.encrypt_int(x, 64, b"film-B")


def test_ff1_rejects_too_short_input():
    with pytest.raises(ValueError):
        FF1(os.urandom(16), radix=2).encrypt([0, 1, 0, 1])


SIP_KEY = bytes(range(16))


@pytest.mark.parametrize("n,out", [
    (0, "310e0edd47db6f72"),
    (1, "fd67dc93c539f874"),
    (15, "e545be4961ca29a1"),
    (63, "724506eb4c328a95"),
])
def test_siphash_kat(n, out):
    assert siphash24_bytes(SIP_KEY, bytes(range(n))) == h(out)


def test_siphash_key_sensitivity():
    assert siphash24(SIP_KEY, b"x") != siphash24(bytes(16), b"x")
