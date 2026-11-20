import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from common import SESSION, USER, available, clip_path, decode_or_none, embed_qim, embed_ss, film_setup
from videowm import attacks
from videowm.dct import block_dct, get_band
from videowm.embed.qim_blind import K, _lattice_dist
from videowm.embed.ss_nonblind import GroupMap, bit_index
from videowm.keys import pattern
from videowm.metrics import psnr
from videowm.video_io import probe, read_frames

WORK = ROOT / "data" / "out" / "exp_nbqim"
WORK.mkdir(parents=True, exist_ok=True)
CLIPS = ["foreman_cif", "akiyo_cif", "bbb_action_720p", "tos_static_720p"]
DELTA, NB = 24.0, 188


def q(p, delta, off):
    return delta * np.round((p - off) / delta) + off


def soft_all(pirate, src, stego_key):
    info = probe(src)
    W, H = info.width, info.height
    gm = GroupMap(stego_key, info.fps)
    nb = (H // 8) * (W // 8)
    out = {k: np.zeros(NB) for k in ("SS non-blind", "QIM blind", "QIM non-blind")}
    cache = (-1, None, None)
    for j, (fo, fp) in enumerate(zip(read_frames(src, W, H), read_frames(pirate, W, H))):
        g = gm.group(j)
        if cache[0] != g:
            cache = (g, bit_index(g, nb, NB), pattern(stego_key, g, nb, K))
        idx, u = cache[1], cache[2]
        co, cp = get_band(block_dct(fo.y)), get_band(block_dct(fp.y))
        out["SS non-blind"] += np.bincount(idx, weights=np.sum((cp - co) * u, axis=1), minlength=NB)
        p = np.sum(cp * u, axis=1) / np.sqrt(K)
        out["QIM blind"] += np.bincount(idx, weights=_lattice_dist(p, DELTA, 0.0) - _lattice_dist(p, DELTA, DELTA / 2), minlength=NB)
        p0 = np.sum(co * u, axis=1) / np.sqrt(K)
        nbq = np.abs(p - q(p0, DELTA, 0.0)) - np.abs(p - q(p0, DELTA, DELTA / 2))
        out["QIM non-blind"] += np.bincount(idx, weights=nbq, minlength=NB)
    return out


def mean_psnr(a, b, step=10):
    return float(np.mean([psnr(x.y, y.y) for i, (x, y) in enumerate(zip(read_frames(a), read_frames(b))) if i % step == 0]))


def main():
    attacks_ = [("CRF 23", lambda m, d: attacks.reencode(m, d, 23)), ("CRF 28", lambda m, d: attacks.reencode(m, d, 28)),
                ("CRF 35", lambda m, d: attacks.reencode(m, d, 35)), ("scale 50 % + CRF 23", lambda m, d: attacks.scale(m, d, 0.5, 23))]
    print("raw bit errors of 188 after soft combining (ID found?)")
    for clip in available(sys.argv[1:] or CLIPS):
        src = clip_path(clip)
        keys, codec = film_setup(clip)
        cw = codec.encode(USER, SESSION)
        ms, mq = WORK / f"{clip}_ss.mkv", WORK / f"{clip}_qim.mkv"
        embed_ss(src, ms, cw, keys)
        embed_qim(src, mq, cw, keys, DELTA)
        print(f"{clip}: PSNR(Y) SS {mean_psnr(src, ms):.2f} dB, QIM {mean_psnr(src, mq):.2f} dB", flush=True)
        for name, atk in attacks_:
            cells = []
            for marked, decs in ((ms, ["SS non-blind"]), (mq, ["QIM blind", "QIM non-blind"])):
                pirate = atk(marked, WORK / f"{clip}_{marked.stem}_atk.mp4")
                soft = soft_all(pirate, src, keys.stego_key)
                for d in decs:
                    for tag, thr in (("", 0.0), (" med", float(np.median(soft[d])))):
                        bits = (soft[d] > thr).astype(np.uint8)
                        ok = decode_or_none(codec, bits) is not None
                        cells.append(f"{d}{tag}: {int(np.sum(bits != cw)):3d} {'OK' if ok else '--'}")
                Path(pirate).unlink()
            print(f"  {name:20s} " + "   ".join(cells), flush=True)
        ms.unlink(); mq.unlink()


if __name__ == "__main__":
    main()
