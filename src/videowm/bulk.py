import secrets

from .ciphers import FF1, PRESENT128


def present_ctr_encrypt(key: bytes, data: bytes) -> bytes:
    nonce = secrets.randbits(64)
    return nonce.to_bytes(8, "big") + PRESENT128(key).ctr(data, nonce)


def present_ctr_decrypt(key: bytes, blob: bytes) -> bytes:
    return PRESENT128(key).ctr(blob[8:], int.from_bytes(blob[:8], "big"))


def _ff1_chunks(key: bytes, data: bytes, decrypt: bool) -> bytes:
    if len(data) % 8:
        raise ValueError("FF1 bulk mode needs a multiple of 8 bytes")
    f = FF1(key, radix=2)
    op = f.decrypt_int if decrypt else f.encrypt_int
    out = bytearray(len(data))
    for i in range(0, len(data), 8):
        x = int.from_bytes(data[i:i + 8], "big")
        out[i:i + 8] = op(x, 64, (i // 8).to_bytes(8, "big")).to_bytes(8, "big")
    return bytes(out)


def ff1_encrypt(key: bytes, data: bytes) -> bytes:
    return _ff1_chunks(key, data, decrypt=False)


def ff1_decrypt(key: bytes, data: bytes) -> bytes:
    return _ff1_chunks(key, data, decrypt=True)
