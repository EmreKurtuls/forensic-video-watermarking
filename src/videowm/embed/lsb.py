import hashlib

import numpy as np

from ..video_io import Frame, VideoWriter, probe, read_frames
from .ss_nonblind import EmbedStats


def capacity_bits(width: int, height: int, frames: int) -> int:
    return width * height * frames


def _directions(stego_key: bytes, n: int) -> np.ndarray:
    raw = hashlib.shake_256(stego_key + b"lsb").digest((n + 7) // 8)
    return np.unpackbits(np.frombuffer(raw, np.uint8))[:n].astype(np.int16) * 2 - 1


def embed_video(src, dst, payload: bytes, stego_key: bytes, codec_args: list[str] | None = None) -> EmbedStats:
    info = probe(src, count_frames=True)
    bits = np.unpackbits(np.frombuffer(payload, np.uint8))
    cap = capacity_bits(info.width, info.height, info.nframes)
    if len(bits) > cap:
        raise ValueError(f"payload needs {len(bits)} bits, video holds {cap}")
    dirs = _directions(stego_key, len(bits))
    per = info.width * info.height
    se, n, pos = 0.0, 0, 0
    with VideoWriter(dst, info.width, info.height, info.fps, codec_args, color=info.color) as wr:
        for fr in read_frames(src, info.width, info.height):
            y = fr.y
            if pos < len(bits):
                take = min(per, len(bits) - pos)
                flat = y.reshape(-1).astype(np.int16)
                seg, b, dr = flat[:take], bits[pos:pos + take], dirs[pos:pos + take]
                mism = (seg & 1) != b
                step = np.where(seg == 0, 1, np.where(seg == 255, -1, dr))
                seg[mism] += step[mism]
                y2 = flat.astype(np.uint8).reshape(y.shape)
                pos += take
            else:
                y2 = y
            se += float(np.mean((y2.astype(np.float64) - y) ** 2))
            n += 1
            wr.write(Frame(y2, fr.u, fr.v))
    return EmbedStats(n, 1, 0, se / max(n, 1))


def extract_bits(marked, nbits: int) -> np.ndarray:
    out, pos = np.empty(nbits, np.uint8), 0
    for fr in read_frames(marked):
        if pos >= nbits:
            break
        take = min(fr.y.size, nbits - pos)
        out[pos:pos + take] = fr.y.reshape(-1)[:take] & 1
        pos += take
    if pos < nbits:
        raise ValueError("video shorter than the payload")
    return out


def extract_bytes(marked, nbytes: int) -> bytes:
    return np.packbits(extract_bits(marked, 8 * nbytes)).tobytes()
