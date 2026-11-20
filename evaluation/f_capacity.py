import csv
import os
import sys

import numpy as np

from common import ALPHA, CLIPS, RESULTS, WORK, available, clip_path, film_setup
from videowm import attacks
from videowm.embed import lsb, qim_blind, ss_nonblind
from videowm.extract import extract_soft
from videowm.metrics import quality
from videowm.video_io import probe

FIELDS = ["clip", "method", "strength", "psnr_y", "ssim_y", "meets_threshold", "raw_bits_per_s", "channel",
          "ber", "bsc_capacity_bits_per_s"]


def h2(p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def main():
    out = open(RESULTS / "f_capacity.csv", "w", newline="")
    w = csv.DictWriter(out, fieldnames=FIELDS)
    w.writeheader()
    rng = np.random.default_rng(3)
    for clip in available(sys.argv[1:] or CLIPS):
        src = clip_path(clip)
        info = probe(src, count_frames=True)
        fps = float(info.fps)
        G = ss_nonblind.group_length(info.fps)
        keys, _ = film_setup(clip)
        ngroups = ss_nonblind.GroupMap(keys.stego_key, info.fps).complete_groups(info.nframes)
        nb = (info.height // 8) * (info.width // 8)
        nbits = nb * ngroups
        bits = rng.integers(0, 2, nbits, dtype=np.uint8)
        for method, strength in (("SS", ALPHA), ("QIM", qim_blind.DELTA_EQUAL_QUALITY), ("LSB", None)):
            m = WORK / f"f_{clip}.mkv"
            if method == "SS":
                ss_nonblind.embed_video(src, m, bits, keys.stego_key, strength)
                read = lambda v: (extract_soft(v, src, keys.stego_key, nbits, aligned=False).soft > 0).astype(np.uint8)
                rate = nb * fps / G
            elif method == "QIM":
                qim_blind.embed_video(src, m, bits, keys.stego_key, strength)
                read = lambda v: (qim_blind.extract_soft(v, keys.stego_key, strength, nbits) > 0).astype(np.uint8)
                rate = nb * fps / G
            else:
                data = rng.integers(0, 256, info.width * info.height * info.nframes // 8, dtype=np.uint8).tobytes()
                lsb.embed_video(src, m, data, keys.stego_key)
                ref = np.unpackbits(np.frombuffer(data, np.uint8))
                read = lambda v: lsb.extract_bits(v, len(ref))
                rate = info.width * info.height * fps
            truth = bits if method != "LSB" else ref
            q = quality(src, m, step=4)
            meets = q["psnr_y"] >= 45 and q["ssim_y"] >= 0.99
            for channel, crf in (("lossless", None), ("H.264 CRF 23", 23), ("H.264 CRF 28", 28)):
                v = m if crf is None else attacks.reencode(m, WORK / f"f_{clip}.mp4", crf)
                ber = float(np.mean(read(v) != truth))
                row = dict(clip=clip, method=method, strength="" if strength is None else round(strength, 2),
                           psnr_y=round(q["psnr_y"], 2), ssim_y=round(q["ssim_y"], 4), meets_threshold=meets,
                           raw_bits_per_s=round(rate), channel=channel, ber=round(ber, 5),
                           bsc_capacity_bits_per_s=round(rate * (1 - h2(ber))))
                w.writerow(row)
                out.flush()
                print(row, flush=True)
                if crf is not None:
                    os.remove(v)
            os.remove(m)
    out.close()


if __name__ == "__main__":
    main()
