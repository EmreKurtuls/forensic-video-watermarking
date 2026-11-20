SBOX = [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]
INV_SBOX = [SBOX.index(x) for x in range(16)]
M64 = (1 << 64) - 1
M128 = (1 << 128) - 1


def _p(i: int) -> int:
    return 63 if i == 63 else (i * 16) % 63


def _p_inv(i: int) -> int:
    return 63 if i == 63 else (i * 4) % 63


def _spread(byte_pos: int, value: int, perm) -> int:
    out = 0
    for b in range(8):
        if (value >> b) & 1:
            out |= 1 << perm(byte_pos * 8 + b)
    return out


_SP = [[_spread(j, SBOX[v & 0xF] | (SBOX[v >> 4] << 4), _p) for v in range(256)] for j in range(8)]
_PINV = [[_spread(j, v, _p_inv) for v in range(256)] for j in range(8)]
_SINV = [INV_SBOX[v & 0xF] | (INV_SBOX[v >> 4] << 4) for v in range(256)]


def _round_keys(key: int) -> list[int]:
    k, rks = key, []
    for rc in range(1, 32):
        rks.append(k >> 64)
        k = ((k << 61) | (k >> 67)) & M128
        top = (SBOX[(k >> 124) & 0xF] << 4) | SBOX[(k >> 120) & 0xF]
        k = (k & ~(0xFF << 120)) | (top << 120)
        k ^= rc << 62
    rks.append(k >> 64)
    return rks


class PRESENT128:
    def __init__(self, key: bytes):
        if len(key) != 16:
            raise ValueError("PRESENT-128 key must be 16 bytes")
        self._rk = _round_keys(int.from_bytes(key, "big"))

    def encrypt_int(self, block: int) -> int:
        T0, T1, T2, T3, T4, T5, T6, T7 = _SP
        rks = self._rk
        s = block & M64
        for rk in rks[:31]:
            s ^= rk
            s = (T0[s & 255] | T1[(s >> 8) & 255] | T2[(s >> 16) & 255] | T3[(s >> 24) & 255]
                 | T4[(s >> 32) & 255] | T5[(s >> 40) & 255] | T6[(s >> 48) & 255] | T7[s >> 56])
        return s ^ rks[31]

    def decrypt_int(self, block: int) -> int:
        P0, P1, P2, P3, P4, P5, P6, P7 = _PINV
        S, rks = _SINV, self._rk
        s = (block & M64) ^ rks[31]
        for rk in reversed(rks[:31]):
            s = (P0[s & 255] | P1[(s >> 8) & 255] | P2[(s >> 16) & 255] | P3[(s >> 24) & 255]
                 | P4[(s >> 32) & 255] | P5[(s >> 40) & 255] | P6[(s >> 48) & 255] | P7[s >> 56])
            s = (S[s & 255] | S[(s >> 8) & 255] << 8 | S[(s >> 16) & 255] << 16 | S[(s >> 24) & 255] << 24
                 | S[(s >> 32) & 255] << 32 | S[(s >> 40) & 255] << 40 | S[(s >> 48) & 255] << 48 | S[s >> 56] << 56)
            s ^= rk
        return s

    def encrypt_block(self, block: bytes) -> bytes:
        return self.encrypt_int(int.from_bytes(_check8(block), "big")).to_bytes(8, "big")

    def decrypt_block(self, block: bytes) -> bytes:
        return self.decrypt_int(int.from_bytes(_check8(block), "big")).to_bytes(8, "big")

    def ctr(self, data: bytes, nonce: int = 0) -> bytes:
        out = bytearray(len(data))
        for i in range(0, len(data), 8):
            ks = self.encrypt_int((nonce + i // 8) & M64).to_bytes(8, "big")
            for j, b in enumerate(data[i:i + 8]):
                out[i + j] = b ^ ks[j]
        return bytes(out)


def _check8(block: bytes) -> bytes:
    if len(block) != 8:
        raise ValueError("PRESENT block must be 8 bytes")
    return block
