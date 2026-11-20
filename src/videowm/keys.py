import hashlib
import secrets
from dataclasses import dataclass

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

MASTER_KEY_LEN = 32


def generate_master_key() -> bytes:
    return secrets.token_bytes(MASTER_KEY_LEN)


def _hkdf(master_key: bytes, info: bytes, length: int) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=None, info=info).derive(master_key)


@dataclass(frozen=True)
class FilmKeys:
    film_id: bytes
    enc_key: bytes
    mac_key: bytes
    stego_key: bytes

    def __repr__(self) -> str:
        return f"FilmKeys(film_id={self.film_id!r}, <keys hidden>)"


def derive_film_keys(master_key: bytes, film_id: bytes) -> FilmKeys:
    if len(master_key) != MASTER_KEY_LEN:
        raise ValueError("master key must be 32 bytes")
    if not film_id:
        raise ValueError("film_id must not be empty")
    return FilmKeys(
        film_id=film_id,
        enc_key=_hkdf(master_key, b"enc" + film_id, 16),
        mac_key=_hkdf(master_key, b"mac" + film_id, 16),
        stego_key=_hkdf(master_key, b"stego" + film_id, 32),
    )


def pattern(stego_key: bytes, group_no: int, nblocks: int, ncoefs: int) -> np.ndarray:
    if not 0 <= group_no < 2 ** 32:
        raise ValueError("group_no out of range")
    n = nblocks * ncoefs
    raw = hashlib.shake_256(stego_key + group_no.to_bytes(4, "big")).digest((n + 7) // 8)
    bits = np.unpackbits(np.frombuffer(raw, np.uint8))[:n]
    return (bits.astype(np.float64) * 2.0 - 1.0).reshape(nblocks, ncoefs)


def save_master_key(path, key: bytes, overwrite: bool = False) -> None:
    import os
    if len(key) != MASTER_KEY_LEN:
        raise ValueError("master key must be 32 bytes")
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if overwrite else os.O_EXCL)
    fd = os.open(path, flags, 0o600)
    try:
        os.write(fd, key)
    finally:
        os.close(fd)
    os.chmod(path, 0o600)


def load_master_key(path) -> bytes:
    with open(path, "rb") as f:
        key = f.read()
    if len(key) != MASTER_KEY_LEN:
        raise ValueError("master key file must hold exactly 32 bytes")
    return key
