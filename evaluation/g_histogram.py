import os

import numpy as np

from common import CLIPS, CSV, SESSION, USER, WORK, available, clip_path, embed_ss, film_setup
from videowm import attacks
from videowm.dct import block_dct, get_band
from videowm.metrics import chi_square, kl_divergence, rgb_histograms
from videowm.video_io import read_frames


def dct_hist(path, step=5, lim=60):
    h = np.zeros(2 * lim + 1, np.int64)
    for i, f in enumerate(read_frames(path)):
        if i % step == 0:
            c = np.clip(np.rint(get_band(block_dct(f.y))).astype(int), -lim, lim) + lim
            h += np.bincount(c.ravel(), minlength=2 * lim + 1)
    return h


def main():
    out = CSV("g_histogram", ["clip", "comparison", "channel", "chi_square", "kl_bits"])
    hists = {}
    for clip in available(CLIPS):
        src = clip_path(clip)
        keys, codec = film_setup(clip)
        marked = WORK / f"g_{clip}.mkv"
        embed_ss(src, marked, codec.encode(USER, SESSION), keys)
        ref = attacks.reencode(src, WORK / f"g_{clip}_crf18.mp4", 18)
        h0, hm, hr = rgb_histograms(src), rgb_histograms(marked), rgb_histograms(ref)
        hists[clip] = (h0, hm, hr)
        for comp, h in (("marked (SS)", hm), ("reference: CRF 18 re-encode", hr)):
            for c, ch in enumerate("RGB"):
                out.row(clip=clip, comparison=comp, channel=ch, chi_square=chi_square(h0[c], h[c]),
                        kl_bits=kl_divergence(h0[c], h[c]))
        d0, dm = dct_hist(src), dct_hist(marked)
        out.row(clip=clip, comparison="marked (SS)", channel="Y low-band DCT", chi_square=chi_square(d0, dm),
                kl_bits=kl_divergence(d0, dm))
        os.remove(marked)
        os.remove(ref)
    np.savez(WORK.parent / "g_histograms.npz", **{f"{k}_{n}": v for k, t in hists.items() for n, v in zip(("orig", "marked", "crf18"), t)})
    out.close()


if __name__ == "__main__":
    main()
