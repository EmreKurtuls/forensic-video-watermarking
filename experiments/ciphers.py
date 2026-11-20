from math import ceil, log2
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

SBOX = [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]
M64 = (1 << 64) - 1

def _p(i):
    return 63 if i == 63 else (i * 16) % 63

SP = []
for byte_pos in range(8):
    t = []
    for v in range(256):
        out = 0
        s = SBOX[v & 0xF] | (SBOX[v >> 4] << 4)
        for b in range(8):
            if (s >> b) & 1:
                out |= 1 << _p(byte_pos * 8 + b)
        t.append(out)
    SP.append(t)

def present128_round_keys(key: int):
    k = key
    rks = []
    for rc in range(1, 32):
        rks.append(k >> 64)
        k = ((k << 61) | (k >> 67)) & ((1 << 128) - 1)
        top = (SBOX[(k >> 124) & 0xF] << 4) | SBOX[(k >> 120) & 0xF]
        k = (k & ~(0xFF << 120)) | (top << 120)
        k ^= rc << 62
    rks.append(k >> 64)
    return rks

def present_encrypt_block(state: int, rks) -> int:
    T0, T1, T2, T3, T4, T5, T6, T7 = SP
    for rk in rks[:31]:
        s = state ^ rk
        state = (T0[s & 255] | T1[(s >> 8) & 255] | T2[(s >> 16) & 255] | T3[(s >> 24) & 255]
                 | T4[(s >> 32) & 255] | T5[(s >> 40) & 255] | T6[(s >> 48) & 255] | T7[(s >> 56) & 255])
    return state ^ rks[31]

def present_ctr(data: bytes, rks, nonce: int = 0) -> bytes:
    out = bytearray()
    for i in range(0, len(data), 8):
        ks = present_encrypt_block((nonce + i // 8) & M64, rks).to_bytes(8, "big")
        chunk = data[i:i + 8]
        out += bytes(a ^ b for a, b in zip(chunk, ks))
    return bytes(out)

class FF1:
    def __init__(self, key: bytes, radix: int):
        self.radix = radix
        self.key = key

    def _ecb(self, data: bytes) -> bytes:
        e = Cipher(algorithms.AES(self.key), modes.ECB()).encryptor()
        return e.update(data) + e.finalize()

    def _prf(self, data: bytes) -> bytes:
        e = Cipher(algorithms.AES(self.key), modes.CBC(b"\x00" * 16)).encryptor()
        return (e.update(data) + e.finalize())[-16:]

    def encrypt_num(self, a: int, b: int, n: int, tweak: bytes = b"") -> tuple:
        r = self.radix
        t = len(tweak)
        u = n // 2
        v = n - u
        bl = ceil(ceil(v * log2(r)) / 8)
        d = 4 * ceil(bl / 4) + 4
        P = bytes([1, 2, 1]) + r.to_bytes(3, "big") + bytes([10, u % 256]) + n.to_bytes(4, "big") + t.to_bytes(4, "big")
        pad = b"\x00" * ((-t - bl - 1) % 16)
        for i in range(10):
            Q = tweak + pad + bytes([i]) + b.to_bytes(bl, "big")
            R = self._prf(P + Q)
            if d > 16:
                ri = int.from_bytes(R, "big")
                extra = b"".join((ri ^ j).to_bytes(16, "big") for j in range(1, ceil(d / 16)))
                S = (R + self._ecb(extra))[:d]
            else:
                S = R[:d]
            y = int.from_bytes(S, "big")
            m = u if i % 2 == 0 else v
            c = (a + y) % (r ** m)
            a, b = b, c
        return a, b

    def encrypt_digits(self, digits: str, tweak: bytes = b"") -> str:
        n = len(digits)
        u = n // 2
        a, b = self.encrypt_num(int(digits[:u], self.radix), int(digits[u:], self.radix), n, tweak)
        return _to_str(a, self.radix, u) + _to_str(b, self.radix, n - u)

    def encrypt_bits(self, x: int, n: int, tweak: bytes = b"") -> int:
        u = n // 2
        v = n - u
        a, b = self.encrypt_num(x >> v, x & ((1 << v) - 1), n, tweak)
        return (a << u) | b if (n - u) == u else (a << (n - u)) | b

def _to_str(x, r, length):
    digits = []
    for _ in range(length):
        digits.append("0123456789abcdefghijklmnopqrstuvwxyz"[x % r])
        x //= r
    return "".join(reversed(digits))

def siphash24(key: bytes, msg: bytes) -> int:
    k0 = int.from_bytes(key[:8], "little"); k1 = int.from_bytes(key[8:], "little")
    v0 = k0 ^ 0x736f6d6570736575; v1 = k1 ^ 0x646f72616e646f6d
    v2 = k0 ^ 0x6c7967656e657261; v3 = k1 ^ 0x7465646279746573
    rotl = lambda x, b: ((x << b) | (x >> (64 - b))) & M64
    def rnd(v0, v1, v2, v3):
        v0 = (v0 + v1) & M64; v1 = rotl(v1, 13); v1 ^= v0; v0 = rotl(v0, 32)
        v2 = (v2 + v3) & M64; v3 = rotl(v3, 16); v3 ^= v2
        v0 = (v0 + v3) & M64; v3 = rotl(v3, 21); v3 ^= v0
        v2 = (v2 + v1) & M64; v1 = rotl(v1, 17); v1 ^= v2; v2 = rotl(v2, 32)
        return v0, v1, v2, v3
    b = len(msg) << 56
    end = len(msg) - len(msg) % 8
    for i in range(0, end, 8):
        m = int.from_bytes(msg[i:i + 8], "little")
        v3 ^= m
        v0, v1, v2, v3 = rnd(v0, v1, v2, v3); v0, v1, v2, v3 = rnd(v0, v1, v2, v3)
        v0 ^= m
    b |= int.from_bytes(msg[end:], "little")
    v3 ^= b
    v0, v1, v2, v3 = rnd(v0, v1, v2, v3); v0, v1, v2, v3 = rnd(v0, v1, v2, v3)
    v0 ^= b
    v2 ^= 0xFF
    for _ in range(4):
        v0, v1, v2, v3 = rnd(v0, v1, v2, v3)
    return v0 ^ v1 ^ v2 ^ v3
