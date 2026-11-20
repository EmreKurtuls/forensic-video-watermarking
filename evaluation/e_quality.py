import os
import sys
from concurrent.futures import ProcessPoolExecutor

import csv
import numpy as np

from common import (CLIPS, RESULTS, SESSION, USER, WORK, Timer, available, clip_path, embed_qim, embed_ss,
                    extract_qim, extract_ss, film_setup)
from videowm import attacks
from videowm.embed import lsb
from videowm.metrics import quality, vmaf
from videowm.video_io import probe

ALPHAS = (1.0, 1.5, 2.0, 3.1, 4.5, 6.0, 9.0)
QIM_DELTAS = (16, 24, 32, 48, 64, 96)
FIELDS = ["clip", "method", "strength", "psnr_y", "psnr_rgb", "ssim_y", "ssim_y_min", "vmaf", "ber_lossless",
          "raw_errors_crf28", "identified_crf28", "encrypt_ms", "embed_s"]


def run_clip(clip):
    src = clip_path(clip)
    info = probe(src, count_frames=True)
    keys, codec = film_setup(clip)
    with Timer() as tenc:
        cw = codec.encode(USER, SESSION)
    rows = []
    settings = [("SS", a) for a in ALPHAS] + [("QIM", d) for d in QIM_DELTAS] + [("LSB", None)]
    for method, k in settings:
        m = WORK / f"e_{clip}.mkv"
        if method == "LSB":
            data = os.urandom(lsb.capacity_bits(info.width, info.height, info.nframes) // 8)
            with Timer() as te:
                lsb.embed_video(src, m, data, keys.stego_key)
            ber0 = float(np.mean(np.unpackbits(np.frombuffer(lsb.extract_bytes(m, len(data)), np.uint8))
                                 != np.unpackbits(np.frombuffer(data, np.uint8))))
            p = attacks.reencode(m, WORK / f"e_{clip}.mp4", 28)
            back = np.unpackbits(np.frombuffer(lsb.extract_bytes(p, len(data)), np.uint8))
            err28, ok28 = int(np.sum(back != np.unpackbits(np.frombuffer(data, np.uint8)))), False
        else:
            with Timer() as te:
                (embed_ss(src, m, cw, keys, k) if method == "SS" else embed_qim(src, m, cw, keys, k))
            ext = (lambda v: extract_ss(v, src, codec)[:2]) if method == "SS" else (lambda v: extract_qim(v, info, codec, k))
            _, b0 = ext(m)
            ber0 = float(np.mean(b0 != cw))
            p = attacks.reencode(m, WORK / f"e_{clip}.mp4", 28)
            ident, b28 = ext(p)
            err28 = int(np.sum(b28 != cw))
            ok28 = ident is not None and ident.user_id == USER
        q = quality(src, m, step=4)
        q["vmaf"] = vmaf(src, m, threads=2)
        rows.append(dict(clip=clip, method=method, strength="" if k is None else k, **{a: round(b, 4) for a, b in q.items()},
                         ber_lossless=round(ber0, 5), raw_errors_crf28=err28, identified_crf28=ok28,
                         encrypt_ms=round(tenc.wall * 1000, 3), embed_s=round(te.wall, 2)))
        print(f"[{clip}] {method} {k}: {q} ber0={ber0:.4f} crf28 err={err28}", flush=True)
        os.remove(p)
        os.remove(m)
    return rows


def main(workers=3):
    clips = available(sys.argv[1:] or CLIPS)
    with ProcessPoolExecutor(workers) as ex, open(RESULTS / "e_quality.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for rows in ex.map(run_clip, clips):
            w.writerows(rows)
            f.flush()


if __name__ == "__main__":
    main()
