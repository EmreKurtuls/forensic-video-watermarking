import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from common import SESSION, available, clip_path, decode_or_none, embed_qim, film_setup
from videowm import attacks
from videowm.embed import qim_blind
from videowm.video_io import probe

WORK = ROOT / "data" / "out" / "exp_qim_thr"
WORK.mkdir(parents=True, exist_ok=True)
CLIPS = ["akiyo_cif", "foreman_cif", "park_joy_720p", "bbb_action_720p", "sintel_dark_720p", "tos_static_720p"]
USERS = [1001, 2002]
DELTAS = [24.0, 64.0]
ATTACKS = [("lossless", None), ("CRF 23", lambda m, d: attacks.reencode(m, d, 23)),
           ("CRF 28", lambda m, d: attacks.reencode(m, d, 28)), ("scale 50% + CRF 23", lambda m, d: attacks.scale(m, d, 0.5, 23))]


def two_means(soft):
    t = float(np.median(soft))
    for _ in range(50):
        hi, lo = soft[soft > t], soft[soft <= t]
        if len(hi) == 0 or len(lo) == 0:
            return t
        t2 = (hi.mean() + lo.mean()) / 2
        if abs(t2 - t) < 1e-9:
            break
        t = t2
    return t


THRESHOLDS = [("zero", lambda s: 0.0), ("median", lambda s: float(np.median(s))), ("2-means", two_means)]


def main():
    clips = available(sys.argv[1:] or CLIPS)
    tally = {}
    for clip in clips:
        src = clip_path(clip)
        info = probe(src)
        for user in USERS:
            keys, codec = film_setup(clip)
            cw = codec.encode(user, SESSION)
            ones = int(cw.sum())
            for delta in DELTAS:
                m = WORK / f"{clip}.mkv"
                embed_qim(src, m, cw, keys, delta)
                for name, atk in ATTACKS:
                    v = m if atk is None else atk(m, WORK / f"{clip}_atk.mp4")
                    soft = qim_blind.extract_soft(v, keys.stego_key, delta, len(cw), fps=info.fps, width=info.width, height=info.height)
                    cells = []
                    for tn, tf in THRESHOLDS:
                        bits = (soft > tf(soft)).astype(np.uint8)
                        err = int(np.sum(bits != cw))
                        ok = decode_or_none(codec, bits) is not None
                        k = (delta, name, tn)
                        tally[k] = tally.get(k, 0) + ok
                        cells.append(f"{tn} {err:3d} {'OK' if ok else '--'}")
                    print(f"{clip:18s} user {user} ones {ones:3d} D={delta:.0f} {name:20s} " + "   ".join(cells), flush=True)
                    if atk is not None:
                        Path(v).unlink()
                m.unlink()
    n = len(clips) * len(USERS)
    print(f"\nidentified, of {n} (clips x users)")
    for delta in DELTAS:
        for name, _ in ATTACKS:
            print(f"D={delta:.0f} {name:20s} " + "   ".join(f"{tn} {tally[(delta, name, tn)]:2d}" for tn, _ in THRESHOLDS))


if __name__ == "__main__":
    main()
