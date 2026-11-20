def _xtime(a: int) -> int:
    a <<= 1
    return a ^ 0x11B if a & 0x100 else a


def _gmul(a: int, b: int) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        a = _xtime(a)
        b >>= 1
    return r


def _build_sbox() -> list[int]:
    sbox = []
    for x in range(256):
        inv = 0
        if x:
            inv = 1
            for _ in range(254):
                inv = _gmul(inv, x)
        s = inv
        for k in range(1, 5):
            s ^= ((inv << k) | (inv >> (8 - k))) & 0xFF
        sbox.append(s ^ 0x63)
    return sbox


SBOX = _build_sbox()
_TE0 = [(_gmul(s, 2) << 24) | (s << 16) | (s << 8) | _gmul(s, 3) for s in SBOX]
_ror8 = lambda w: ((w >> 8) | (w << 24)) & 0xFFFFFFFF
_TE1 = [_ror8(w) for w in _TE0]
_TE2 = [_ror8(w) for w in _TE1]
_TE3 = [_ror8(w) for w in _TE2]
_RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]


class AES128:

    def __init__(self, key: bytes):
        if len(key) != 16:
            raise ValueError("AES-128 key must be 16 bytes")
        w = [int.from_bytes(key[4 * i:4 * i + 4], "big") for i in range(4)]
        for i in range(4, 44):
            t = w[i - 1]
            if i % 4 == 0:
                t = ((t << 8) | (t >> 24)) & 0xFFFFFFFF
                t = (SBOX[t >> 24] << 24) | (SBOX[(t >> 16) & 255] << 16) | \
                    (SBOX[(t >> 8) & 255] << 8) | SBOX[t & 255]
                t ^= _RCON[i // 4 - 1] << 24
            w.append(w[i - 4] ^ t)
        self._w = w

    def encrypt_int(self, block: int) -> int:
        w, T0, T1, T2, T3, S = self._w, _TE0, _TE1, _TE2, _TE3, SBOX
        s0 = ((block >> 96) & 0xFFFFFFFF) ^ w[0]
        s1 = ((block >> 64) & 0xFFFFFFFF) ^ w[1]
        s2 = ((block >> 32) & 0xFFFFFFFF) ^ w[2]
        s3 = (block & 0xFFFFFFFF) ^ w[3]
        for r in range(4, 40, 4):
            t0 = T0[s0 >> 24] ^ T1[(s1 >> 16) & 255] ^ T2[(s2 >> 8) & 255] ^ T3[s3 & 255] ^ w[r]
            t1 = T0[s1 >> 24] ^ T1[(s2 >> 16) & 255] ^ T2[(s3 >> 8) & 255] ^ T3[s0 & 255] ^ w[r + 1]
            t2 = T0[s2 >> 24] ^ T1[(s3 >> 16) & 255] ^ T2[(s0 >> 8) & 255] ^ T3[s1 & 255] ^ w[r + 2]
            t3 = T0[s3 >> 24] ^ T1[(s0 >> 16) & 255] ^ T2[(s1 >> 8) & 255] ^ T3[s2 & 255] ^ w[r + 3]
            s0, s1, s2, s3 = t0, t1, t2, t3
        o0 = (S[s0 >> 24] << 24 | S[(s1 >> 16) & 255] << 16 | S[(s2 >> 8) & 255] << 8 | S[s3 & 255]) ^ w[40]
        o1 = (S[s1 >> 24] << 24 | S[(s2 >> 16) & 255] << 16 | S[(s3 >> 8) & 255] << 8 | S[s0 & 255]) ^ w[41]
        o2 = (S[s2 >> 24] << 24 | S[(s3 >> 16) & 255] << 16 | S[(s0 >> 8) & 255] << 8 | S[s1 & 255]) ^ w[42]
        o3 = (S[s3 >> 24] << 24 | S[(s0 >> 16) & 255] << 16 | S[(s1 >> 8) & 255] << 8 | S[s2 & 255]) ^ w[43]
        return (o0 << 96) | (o1 << 64) | (o2 << 32) | o3

    def encrypt_block(self, block: bytes) -> bytes:
        if len(block) != 16:
            raise ValueError("AES block must be 16 bytes")
        return self.encrypt_int(int.from_bytes(block, "big")).to_bytes(16, "big")
