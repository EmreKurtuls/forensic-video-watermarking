import hashlib
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "experiments"))
from common import CLIPS, SESSION, USER, available, clip_path, decode_or_none, film_setup
from exp_steganalysis import features, temporal_stat
from videowm import attacks
from videowm.align import align, thumbnail
from videowm.dct import LOW_BAND
from videowm.embed.ss_nonblind import bit_index, embed_luma, group_length
from videowm.extract import block_correlation
from videowm.keys import pattern
from videowm.video_io import Frame, VideoWriter, probe, read_frames

WORK = ROOT / "data" / "out" / "exp_jitter"
WORK.mkdir(parents=True, exist_ok=True)
MAX_FRAMES = 200_000


def group_map(stego_key: bytes, G: int, jitter: bool) -> np.ndarray:
    if not jitter:
        return np.arange(MAX_FRAMES) // G
    J = round(G / 4)
    b = np.frombuffer(hashlib.shake_256(stego_key + b"groups").digest(MAX_FRAMES // (G - J) + 1), np.uint8)
    lengths = G - J + (b.astype(int) % (2 * J + 1))
    return np.repeat(np.arange(len(lengths)), lengths)[:MAX_FRAMES]


def embed(src, dst, codeword, stego_key, jitter, alpha=None):
    info = probe(src)
    gm = group_map(stego_key, group_length(info.fps), jitter)
    nb = (info.height // 8) * (info.width // 8)
    bits = np.asarray(codeword, np.uint8)
    cache = (-1, None, None)
    with VideoWriter(dst, info.width, info.height, info.fps, color=info.color) as wr:
        for i, fr in enumerate(read_frames(src, info.width, info.height)):
            g = int(gm[i])
            if cache[0] != g:
                cache = (g, bits[bit_index(g, nb, len(bits))] * 2.0 - 1.0, pattern(stego_key, g, nb, len(LOW_BAND)))
            y2 = embed_luma(fr.y, cache[1], cache[2]) if alpha is None else embed_luma(fr.y, cache[1], cache[2], alpha)
            wr.write(Frame(y2, fr.u, fr.v))
    return gm


def extract_bits(pirate, original, stego_key, jitter, nbits=188):
    info = probe(original)
    W, H = info.width, info.height
    gm = group_map(stego_key, group_length(info.fps), jitter)
    nb = (H // 8) * (W // 8)
    match = align([thumbnail(f.y) for f in read_frames(pirate, W, H)], [thumbnail(f.y) for f in read_frames(original, W, H)])
    soft = np.zeros(nbits)
    it = enumerate(read_frames(original, W, H))
    j, fo = next(it)
    cache = (-1, None, None)
    for k, fp in enumerate(read_frames(pirate, W, H)):
        if k >= len(match):
            break
        while j < match[k]:
            j, fo = next(it)
        g = int(gm[j])
        if cache[0] != g:
            cache = (g, bit_index(g, nb, nbits), pattern(stego_key, g, nb, len(LOW_BAND)))
        soft += np.bincount(cache[1], weights=block_correlation(fp.y, fo.y, cache[2]), minlength=nbits)
    return (soft > 0).astype(np.uint8)


def norm_r(d):
    r = np.full(len(d), np.nan)
    r[2:-1] = d[2:-1] / np.maximum((d[1:-2] + d[3:]) / 2, 1e-9)
    return r


def detect(src=ROOT / "data/raw/BigBuckBunny_320x180.mp4"):
    G = group_length(probe(src).fps)
    keys, codec = film_setup("bbb_full")
    cw = codec.encode(USER, SESSION)
    print("== detection on the full film (phase 0 vs phase 12 = analyst; boundary vs rest = oracle with key) ==")
    for jitter in (False, True):
        tag = "jitter" if jitter else "fixed"
        marked = WORK / f"bbb_full_{tag}.mkv"
        gm = embed(src, marked, cw, keys.stego_key, jitter)
        variants = {"lossless": marked}
        for crf in (18, 23):
            variants[f"CRF {crf}"] = attacks.reencode(marked, WORK / f"bbb_full_{tag}_crf{crf}.mp4", crf) or WORK / f"bbb_full_{tag}_crf{crf}.mp4"
        for name, p in variants.items():
            d = features(p)["d"]
            r, t = norm_r(d), np.arange(len(d))
            ok = ~np.isnan(r)
            p_an = mannwhitneyu(r[(t % G == 0) & ok], r[(t % G == G // 2) & ok], alternative="greater").pvalue
            S, peak, _ = temporal_stat(d, G)
            bnd = np.zeros(len(d), bool)
            bnd[1:] = gm[1:len(d)] != gm[:len(d) - 1]
            p_or = mannwhitneyu(r[bnd & ok], r[~bnd & ok], alternative="greater").pvalue
            print(f"  {tag:6s} {name:8s} analyst p={p_an:.2g}  S={S:.3f} peak={peak:2d}   oracle p={p_or:.2g}", flush=True)
        for p in variants.values():
            Path(p).unlink()


def robust():
    print("== robustness: raw bit errors of 188 (ID found?) ==")
    for clip in available(CLIPS):
        src = clip_path(clip)
        keys, codec = film_setup(clip)
        cw = codec.encode(USER, SESSION)
        for jitter in (False, True):
            tag = "jitter" if jitter else "fixed"
            marked = WORK / f"{clip}_{tag}.mkv"
            embed(src, marked, cw, keys.stego_key, jitter)
            cells = []
            for crf in (23, 28, 35):
                q = WORK / f"{clip}_{tag}_crf{crf}.mp4"
                attacks.reencode(marked, q, crf)
                bits = extract_bits(q, src, keys.stego_key, jitter)
                ok = decode_or_none(codec, bits) is not None
                cells.append(f"CRF{crf}: {int(np.sum(bits != cw)):3d} {'OK ' if ok else 'NO '}")
                q.unlink()
            marked.unlink()
            print(f"  {clip:18s} {tag:6s} " + "  ".join(cells), flush=True)


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("detect", "all"):
        detect()
    if part in ("robust", "all"):
        robust()
