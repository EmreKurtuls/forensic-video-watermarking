from dataclasses import dataclass

import numpy as np

from .align import align, thumbnail
from .dct import LOW_BAND, block_dct, get_band
from .embed.ss_nonblind import GroupMap, bit_index
from .keys import pattern
from .payload import CODEWORD_BITS, Identity, NoValidMark, PayloadCodec
from .video_io import probe, read_frames


@dataclass
class Extraction:
    soft: np.ndarray
    match: list[int]
    pirate_frames: int
    per_frame: np.ndarray | None = None

    @property
    def bits(self) -> np.ndarray:
        return (self.soft > 0).astype(np.uint8)


def block_correlation(y_pirate: np.ndarray, y_orig: np.ndarray, u: np.ndarray) -> np.ndarray:
    diff = get_band(block_dct(y_pirate)) - get_band(block_dct(y_orig))
    return np.sum(diff * u, axis=1)


def extract_soft(pirate, original, stego_key: bytes, nbits: int = CODEWORD_BITS, aligned: bool = True,
                 keep_per_frame: bool = False) -> Extraction:
    info = probe(original)
    W, H = info.width, info.height
    gm = GroupMap(stego_key, info.fps)
    nb = (H // 8) * (W // 8)
    if aligned:
        o_th = [thumbnail(f.y) for f in read_frames(original, W, H)]
        p_th = [thumbnail(f.y) for f in read_frames(pirate, W, H)]
        match = align(p_th, o_th)
    else:
        n_o = sum(1 for _ in read_frames(original, W, H))
        n_p = sum(1 for _ in read_frames(pirate, W, H))
        match = list(range(min(n_o, n_p)))

    soft = np.zeros(nbits)
    per = [] if keep_per_frame else None
    orig_iter = enumerate(read_frames(original, W, H))
    j, fo = next(orig_iter)
    cache = (-1, None, None)
    for k, fp in enumerate(read_frames(pirate, W, H)):
        if k >= len(match):
            break
        while j < match[k]:
            j, fo = next(orig_iter)
        g = gm.group(j)
        if cache[0] != g:
            cache = (g, bit_index(g, nb, nbits), pattern(stego_key, g, nb, len(LOW_BAND)))
        contrib = np.bincount(cache[1], weights=block_correlation(fp.y, fo.y, cache[2]), minlength=nbits)
        soft += contrib
        if per is not None:
            per.append(contrib)
    return Extraction(soft, match, len(match), None if per is None else np.array(per))


def identify(pirate, original, codec: PayloadCodec, aligned: bool = True,
             keep_per_frame: bool = False) -> tuple[Identity | None, Extraction]:
    ex = extract_soft(pirate, original, codec.keys.stego_key, CODEWORD_BITS, aligned, keep_per_frame)
    try:
        return codec.decode(ex.bits), ex
    except NoValidMark:
        return None, ex


def frames_to_identify(per_frame: np.ndarray, codec: PayloadCodec, expected: tuple[int, int], step: int = 1) -> int | None:
    cum = np.cumsum(per_frame, axis=0)
    first_ok = None
    for k in range(step - 1, len(cum), step):
        try:
            ident = codec.decode((cum[k] > 0).astype(np.uint8))
            ok = (ident.user_id, ident.session) == expected
        except NoValidMark:
            ok = False
        if ok and first_ok is None:
            first_ok = k + 1
        elif not ok:
            first_ok = None
    return first_ok
