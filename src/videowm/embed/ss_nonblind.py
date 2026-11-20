from dataclasses import dataclass
import hashlib
from fractions import Fraction

import numpy as np

from ..dct import LOW_BAND, block_dct, block_idct, get_band, set_band
from ..keys import pattern
from ..video_io import Frame, VideoWriter, probe, read_frames

DEFAULT_ALPHA = 2.0 * np.sqrt(12 / len(LOW_BAND))
MASK_LO, MASK_HI = 0.5, 2.0
JITTER = 0.25


def group_length(fps) -> int:
    return max(1, round(float(Fraction(fps))))


class GroupMap:

    def __init__(self, stego_key: bytes, fps):
        self.nominal = group_length(fps)
        j = round(JITTER * self.nominal)
        self.lo = max(1, self.nominal - j)
        self.span = self.nominal + j - self.lo + 1
        if self.span > 256:
            raise ValueError("frame rate too high for one-byte group lengths")
        self._seed = stego_key + b"groups"
        self._ends = np.zeros(0, np.int64)
        self._nbytes = 0

    def _extend(self, frame: int) -> None:
        n = max(256, 2 * self._nbytes)
        limit = 256 - 256 % self.span
        while True:
            b = np.frombuffer(hashlib.shake_256(self._seed).digest(n), np.uint8)
            b = b[b < limit].astype(np.int64)
            self._ends = np.cumsum(self.lo + b % self.span)
            self._nbytes = n
            if len(self._ends) and self._ends[-1] > frame:
                return
            n *= 2

    def group(self, frame: int) -> int:
        if frame < 0:
            raise ValueError("frame must be >= 0")
        if not len(self._ends) or frame >= self._ends[-1]:
            self._extend(frame)
        return int(np.searchsorted(self._ends, frame, side="right"))

    def complete_groups(self, nframes: int) -> int:
        return self.group(nframes) if nframes > 0 else 0


def bit_index(group: int, nblocks: int, nbits: int) -> np.ndarray:
    return (group * nblocks + np.arange(nblocks)) % nbits


def masked_alpha(d: np.ndarray, alpha: float, lo: float = MASK_LO, hi: float = MASK_HI) -> np.ndarray:
    ac = np.sqrt(np.maximum(np.sum(d ** 2, axis=(2, 3)) - d[..., 0, 0] ** 2, 0)).reshape(-1) + 1e-6
    s = np.clip(ac / np.median(ac), lo, hi)
    return alpha * s / np.sqrt(np.mean(s ** 2))


def embed_luma(y: np.ndarray, signs: np.ndarray, u: np.ndarray, alpha: float = DEFAULT_ALPHA) -> np.ndarray:
    d = block_dct(y)
    a = masked_alpha(d, alpha)
    return block_idct(set_band(d, get_band(d) + (a * signs)[:, None] * u), y)


@dataclass
class EmbedStats:
    frames: int
    group_len: int
    nblocks: int
    mse_y: float

    @property
    def psnr_y(self) -> float:
        return 99.0 if self.mse_y == 0 else float(10 * np.log10(255 ** 2 / self.mse_y))


def embed_video(src, dst, codeword: np.ndarray, stego_key: bytes, alpha: float = DEFAULT_ALPHA,
                codec_args: list[str] | None = None, progress=None) -> EmbedStats:
    info = probe(src)
    bits = np.asarray(codeword, np.uint8)
    G = group_length(info.fps)
    gm = GroupMap(stego_key, info.fps)
    bh, bw = info.height // 8, info.width // 8
    nb = bh * bw
    se, n = 0.0, 0
    cache = (-1, None, None)
    with VideoWriter(dst, info.width, info.height, info.fps, codec_args, color=info.color) as wr:
        for i, fr in enumerate(read_frames(src, info.width, info.height)):
            g = gm.group(i)
            if cache[0] != g:
                cache = (g, bits[bit_index(g, nb, len(bits))] * 2.0 - 1.0, pattern(stego_key, g, nb, len(LOW_BAND)))
            y2 = embed_luma(fr.y, cache[1], cache[2], alpha)
            se += float(np.mean((y2.astype(np.float64) - fr.y) ** 2))
            n += 1
            wr.write(Frame(y2, fr.u, fr.v))
            if progress is not None:
                progress(n)
    return EmbedStats(n, G, nb, se / max(n, 1))
