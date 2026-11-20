import hashlib
import hmac
from dataclasses import dataclass
from enum import IntEnum

import bchlib
import numpy as np

from .ciphers import FF1, PRESENT128, siphash24
from .keys import FilmKeys

VERSION = 1
DATA_BITS = 96
CODEWORD_BITS = 188
PARITY_BITS = CODEWORD_BITS - DATA_BITS
BCH_T = 12


class Algorithm(IntEnum):
    PRESENT128 = 1
    FF1 = 2


class NoValidMark(Exception):

    def __init__(self):
        super().__init__("no valid watermark found")


@dataclass(frozen=True)
class Identity:
    user_id: int
    session: int
    corrected_bits: int


def pack_block(user_id: int, session: int, version: int = VERSION) -> int:
    if not 0 <= user_id < 2 ** 32:
        raise ValueError("user_id must fit in 32 bits")
    if not 0 <= session < 2 ** 24:
        raise ValueError("session must fit in 24 bits")
    if not 0 <= version < 16:
        raise ValueError("version must fit in 4 bits")
    return (version << 60) | (user_id << 28) | (session << 4)


def unpack_block(block: int) -> tuple[int, int, int, int]:
    return block >> 60, (block >> 28) & 0xFFFFFFFF, (block >> 4) & 0xFFFFFF, block & 0xF


class PayloadCodec:

    def __init__(self, keys: FilmKeys, algorithm: Algorithm):
        self.keys = keys
        self.algorithm = Algorithm(algorithm)
        if self.algorithm is Algorithm.PRESENT128:
            c = PRESENT128(keys.enc_key)
            self._enc, self._dec = c.encrypt_int, c.decrypt_int
        else:
            f, tw = FF1(keys.enc_key, radix=2), keys.film_id
            self._enc = lambda x: f.encrypt_int(x, 64, tw)
            self._dec = lambda x: f.decrypt_int(x, 64, tw)
        self._kc = hashlib.sha256(keys.enc_key).digest()[:8]
        self._bch = bchlib.BCH(BCH_T, m=8)
        assert self._bch.ecc_bits == PARITY_BITS and self._bch.ecc_bytes == 12

    def _tag(self, c: bytes) -> bytes:
        msg = bytes([self.algorithm]) + self.keys.film_id + self._kc + c
        return (siphash24(self.keys.mac_key, msg) & 0xFFFFFFFF).to_bytes(4, "big")

    def encode_bytes(self, user_id: int, session: int) -> bytes:
        c = self._enc(pack_block(user_id, session)).to_bytes(8, "big")
        p = c + self._tag(c)
        return p + self._bch.encode(p)

    def encode(self, user_id: int, session: int) -> np.ndarray:
        return np.unpackbits(np.frombuffer(self.encode_bytes(user_id, session), np.uint8))[:CODEWORD_BITS]

    def decode(self, bits) -> Identity:
        bits = np.asarray(bits, dtype=np.uint8)
        if bits.shape != (CODEWORD_BITS,) or bits.max(initial=0) > 1:
            raise ValueError("expected 188 bits of 0/1")
        cw = np.packbits(np.concatenate([bits, np.zeros(4, np.uint8)])).tobytes()
        data, ecc = bytearray(cw[:12]), bytearray(cw[12:])
        nerr = self._bch.decode(bytes(data), bytes(ecc))
        if nerr < 0:
            raise NoValidMark()
        self._bch.correct(data, ecc)
        c, tag = bytes(data[:8]), bytes(data[8:])
        if not hmac.compare_digest(tag, self._tag(c)):
            raise NoValidMark()
        version, user_id, session, reserved = unpack_block(self._dec(int.from_bytes(c, "big")))
        if version != VERSION or reserved != 0:
            raise NoValidMark()
        return Identity(user_id, session, nerr)
