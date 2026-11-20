import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import (ALPHA, CLIPS, DELTAS, RESULTS, SESSION, USER, WORK, available, clip_path, embed_qim, embed_ss,
                    extract_lsb, extract_qim, extract_ss, film_setup, lsb_payload)
from videowm import attacks
from videowm.embed import lsb
from videowm.extract import frames_to_identify
from videowm.video_io import probe

FIELDS = ["clip", "method", "attack", "raw_bit_errors", "raw_ber", "identified", "seconds_to_identify", "note"]


def attack_list(info):
    fps = round(float(info.fps))
    A = []
    for crf in (18, 23, 28, 35, 40):
        A.append((f"H.264 CRF {crf}", lambda s, d, c=crf: attacks.reencode(s, d, c, "h264")))
    for crf in (18, 23, 28, 35, 40):
        A.append((f"H.265 CRF {crf}", lambda s, d, c=crf: attacks.reencode(s, d, c, "h265")))
    for f in (0.75, 0.5):
        A.append((f"scale {int(f * 100)}% + CRF 23", lambda s, d, f=f: attacks.scale(s, d, f)))
    A.append(("trim 1 s + drop every 10th + CRF 23",
              lambda s, d: attacks.trim_and_drop(s, d, info.nframes, start=fps, drop_every=10)[0]))
    for burst in (1, 10):
        for rate in (0.001, 0.01, 0.05):
            kind = "random" if burst == 1 else f"burst {burst}"
            A.append((f"TS loss {rate * 100:g}% {kind}",
                      lambda s, d, r=rate, b=burst: attacks.packet_loss(s, d, r, b, seed=7)[0]))
    return A


def run_clip(clip):
    src = clip_path(clip)
    info = probe(src, count_frames=True)
    keys, codec = film_setup(clip)
    cw = codec.encode(USER, SESSION)
    frame_bytes = info.width * info.height // 8
    marked = {}
    marked["SS a=3.1"] = WORK / f"b_{clip}_ss.mkv"
    embed_ss(src, marked["SS a=3.1"], cw, keys, ALPHA)
    for name, d in DELTAS.items():
        marked[name] = WORK / f"b_{clip}_q{int(d)}.mkv"
        embed_qim(src, marked[name], cw, keys, d)
    marked["LSB"] = WORK / f"b_{clip}_lsb.mkv"
    lsb.embed_video(src, marked["LSB"], lsb_payload(codec, frame_bytes), keys.stego_key)

    rows = []
    for attack, fn in [("none (lossless)", None)] + attack_list(info):
        for method, mpath in marked.items():
            pir = mpath if fn is None else fn(mpath, WORK / f"b_{clip}_pirate.mkv" if "TS" in attack else WORK / f"b_{clip}_pirate.mp4")
            note, secs = "", ""
            if not os.path.exists(pir) or os.path.getsize(pir) == 0:
                rows.append(dict(clip=clip, method=method, attack=attack, raw_bit_errors="", raw_ber="", identified=False,
                                 seconds_to_identify="", note="pirate copy not decodable"))
                continue
            try:
                if method.startswith("SS"):
                    ident, bits, ex = extract_ss(pir, src, codec, keep_per_frame=True)
                    n = frames_to_identify(ex.per_frame, codec, (USER, SESSION))
                    secs = "" if n is None else round(n / float(info.fps), 2)
                elif method.startswith("QIM"):
                    ident, bits = extract_qim(pir, info, codec, DELTAS[method])
                else:
                    ident, bits = extract_lsb(pir, codec, frame_bytes)
            except Exception as e:
                ident, bits, note = None, np.full(188, 2, np.uint8), f"extraction failed: {type(e).__name__}"
            err = int(np.sum(bits != cw))
            ok = ident is not None and (ident.user_id, ident.session) == (USER, SESSION)
            rows.append(dict(clip=clip, method=method, attack=attack, raw_bit_errors=err, raw_ber=round(err / 188, 4),
                             identified=ok, seconds_to_identify=secs, note=note))
            print(f"[{clip}] {method:9s} {attack:36s} err={err:3d} {'✅' if ok else '❌'} {secs}", flush=True)
            if fn is not None and os.path.exists(pir):
                os.remove(pir)
                if os.path.exists(str(pir) + ".ts"):
                    os.remove(str(pir) + ".ts")
    for p in marked.values():
        os.remove(p)
    return rows


def main(workers=3):
    clips = available(sys.argv[1:] or CLIPS)
    with ProcessPoolExecutor(workers) as ex, open(RESULTS / "b_degradation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for rows in ex.map(run_clip, clips):
            w.writerows(rows)
            f.flush()


if __name__ == "__main__":
    main()
