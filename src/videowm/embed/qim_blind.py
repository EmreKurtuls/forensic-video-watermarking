import numpy as np

from ..dct import LOW_BAND, block_dct, block_idct, get_band, set_band
from ..keys import pattern
from ..video_io import Frame, VideoWriter, probe, read_frames
from .ss_nonblind import EmbedStats, GroupMap, bit_index, group_length

K = len(LOW_BAND)
DELTA_EQUAL_QUALITY = 24.0
DELTA_EQUAL_ROBUSTNESS = 64.0


def _lattice_dist(p: np.ndarray, delta: float, offset: float) -> np.ndarray:
    return np.abs(p - (delta * np.round((p - offset) / delta) + offset))


def embed_luma(y: np.ndarray, bits01: np.ndarray, u: np.ndarray, delta: float) -> np.ndarray:
    d = block_dct(y)
    c = get_band(d)
    p = np.sum(c * u, axis=1) / np.sqrt(K)
    off = bits01 * (delta / 2)
    q = delta * np.round((p - off) / delta) + off
    return block_idct(set_band(d, c + ((q - p) / np.sqrt(K))[:, None] * u), y)


def soft_luma(y: np.ndarray, u: np.ndarray, delta: float) -> np.ndarray:
    p = np.sum(get_band(block_dct(y)) * u, axis=1) / np.sqrt(K)
    return _lattice_dist(p, delta, 0.0) - _lattice_dist(p, delta, delta / 2)


def embed_video(src, dst, codeword: np.ndarray, stego_key: bytes, delta: float,
                codec_args: list[str] | None = None) -> EmbedStats:
    info = probe(src)
    bits = np.asarray(codeword, np.uint8)
    G = group_length(info.fps)
    gm = GroupMap(stego_key, info.fps)
    nb = (info.height // 8) * (info.width // 8)
    se, n, cache = 0.0, 0, (-1, None, None)
    with VideoWriter(dst, info.width, info.height, info.fps, codec_args, color=info.color) as wr:
        for i, fr in enumerate(read_frames(src, info.width, info.height)):
            g = gm.group(i)
            if cache[0] != g:
                cache = (g, bits[bit_index(g, nb, len(bits))].astype(np.float64), pattern(stego_key, g, nb, K))
            y2 = embed_luma(fr.y, cache[1], cache[2], delta)
            se += float(np.mean((y2.astype(np.float64) - fr.y) ** 2))
            n += 1
            wr.write(Frame(y2, fr.u, fr.v))
    return EmbedStats(n, G, nb, se / max(n, 1))


def extract_soft(marked, stego_key: bytes, delta: float, nbits: int, fps=None, width=None, height=None) -> np.ndarray:
    info = probe(marked)
    W, H = width or info.width, height or info.height
    gm = GroupMap(stego_key, fps if fps is not None else info.fps)
    nb = (H // 8) * (W // 8)
    soft, cache = np.zeros(nbits), (-1, None, None)
    for i, fr in enumerate(read_frames(marked, W, H)):
        g = gm.group(i)
        if cache[0] != g:
            cache = (g, bit_index(g, nb, nbits), pattern(stego_key, g, nb, K))
        np.add.at(soft, cache[1], soft_luma(fr.y, cache[2], delta))
    return soft


def threshold(soft: np.ndarray, iterations: int = 50) -> float:
    t = float(np.median(soft))
    for _ in range(iterations):
        hi, lo = soft[soft > t], soft[soft <= t]
        if len(hi) == 0 or len(lo) == 0:
            break
        t2 = float(hi.mean() + lo.mean()) / 2
        if abs(t2 - t) < 1e-9:
            break
        t = t2
    return t


def decide(soft: np.ndarray) -> np.ndarray:
    return (soft > threshold(soft)).astype(np.uint8)
