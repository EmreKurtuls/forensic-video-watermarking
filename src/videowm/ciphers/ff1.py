from math import ceil

from .aes import AES128

_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


class FF1:
    def __init__(self, key: bytes, radix: int = 2, max_tweak_len: int = 256):
        if not 2 <= radix <= 2 ** 16:
            raise ValueError("radix must be in [2, 2^16]")
        self._aes = AES128(key)
        self.radix = radix
        self.max_tweak_len = max_tweak_len
        self.minlen = 2
        while radix ** self.minlen < 1_000_000:
            self.minlen += 1

    def _prf(self, data: bytes) -> int:
        y = 0
        enc = self._aes.encrypt_int
        for i in range(0, len(data), 16):
            y = enc(y ^ int.from_bytes(data[i:i + 16], "big"))
        return y

    def _feistel(self, digits: list[int], tweak: bytes, decrypt: bool) -> list[int]:
        r, n, t = self.radix, len(digits), len(tweak)
        if not self.minlen <= n <= 2 ** 32:
            raise ValueError(f"input length must be at least {self.minlen} digits for radix {r}")
        if t > self.max_tweak_len:
            raise ValueError("tweak too long")
        if any(not 0 <= x < r for x in digits):
            raise ValueError("digit out of range for radix")
        u, v = n // 2, n - n // 2
        a, b = _num(digits[:u], r), _num(digits[u:], r)
        bl = ceil((r ** v - 1).bit_length() / 8)
        d = 4 * ceil(bl / 4) + 4
        P = bytes([1, 2, 1]) + r.to_bytes(3, "big") + bytes([10, u % 256]) + n.to_bytes(4, "big") + t.to_bytes(4, "big")
        head = P + tweak + b"\x00" * ((-t - bl - 1) % 16)
        mod_u, mod_v = r ** u, r ** v
        rounds = range(9, -1, -1) if decrypt else range(10)
        for i in rounds:
            x = a if decrypt else b
            R = self._prf(head + bytes([i]) + x.to_bytes(bl, "big"))
            S = R.to_bytes(16, "big")
            for j in range(1, ceil(d / 16)):
                S += self._aes.encrypt_int(R ^ j).to_bytes(16, "big")
            y = int.from_bytes(S[:d], "big")
            m = mod_u if i % 2 == 0 else mod_v
            if decrypt:
                a, b = (b - y) % m, a
            else:
                a, b = b, (a + y) % m
        return _str(a, r, u) + _str(b, r, v)

    def encrypt(self, digits: list[int], tweak: bytes = b"") -> list[int]:
        return self._feistel(list(digits), tweak, decrypt=False)

    def decrypt(self, digits: list[int], tweak: bytes = b"") -> list[int]:
        return self._feistel(list(digits), tweak, decrypt=True)

    def encrypt_str(self, s: str, tweak: bytes = b"") -> str:
        return _from_digits(self.encrypt(_to_digits(s), tweak))

    def decrypt_str(self, s: str, tweak: bytes = b"") -> str:
        return _from_digits(self.decrypt(_to_digits(s), tweak))

    def encrypt_int(self, x: int, nbits: int, tweak: bytes = b"") -> int:
        return _num(self.encrypt(_bits(x, nbits), tweak), 2)

    def decrypt_int(self, x: int, nbits: int, tweak: bytes = b"") -> int:
        return _num(self.decrypt(_bits(x, nbits), tweak), 2)


def _num(digits: list[int], r: int) -> int:
    x = 0
    for dgt in digits:
        x = x * r + dgt
    return x


def _str(x: int, r: int, length: int) -> list[int]:
    out = [0] * length
    for i in range(length - 1, -1, -1):
        x, out[i] = divmod(x, r)
    return out


def _bits(x: int, nbits: int) -> list[int]:
    if not 0 <= x < (1 << nbits):
        raise ValueError("value does not fit in nbits")
    return [(x >> (nbits - 1 - i)) & 1 for i in range(nbits)]


def _to_digits(s: str) -> list[int]:
    return [_ALPHABET.index(c) for c in s]


def _from_digits(d: list[int]) -> str:
    return "".join(_ALPHABET[x] for x in d)
