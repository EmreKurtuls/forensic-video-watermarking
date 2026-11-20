import json
import os
import subprocess
import sys

import numpy as np

from videowm.video_io import probe, read_frames

RAW, OUT = "data/raw", "data/clips"
LEN, SKIP, PROXY_FPS = 10, 60, 4

FILMS = {
    "sintel_dark_720p": (f"{RAW}/sintel720/Sintel.2010.720p.mkv", "dark"),
    "tos_static_720p": (f"{RAW}/tos720/tears_of_steel_720p.mov", "static"),
    "bbb_action_720p": (f"{RAW}/bbb720/big_buck_bunny_720p_h264.mov", "action"),
}
XIPH = {
    "akiyo_cif": (f"{RAW}/akiyo_cif.y4m", None),
    "foreman_cif": (f"{RAW}/foreman_cif.y4m", None),
    "park_joy_720p": (f"{RAW}/park_joy_720p50.y4m", None),
    "crowd_run_1080p": (f"{RAW}/crowd_run_1080p50.y4m", None),
}


def proxy_stats(path):
    per_sec, prev, cur = [], None, []
    for i, f in enumerate(read_frames(path, 160, 90, vf=f"fps={PROXY_FPS}", passthrough=False)):
        y = f.y.astype(np.float64)
        mot = 0.0 if prev is None else float(np.mean(np.abs(y - prev)))
        tex = float(np.mean(np.abs(np.diff(y, axis=0))) + np.mean(np.abs(np.diff(y, axis=1))))
        cur.append((y.mean(), mot, tex, float(np.mean(y < 20))))
        prev = y
        if len(cur) == PROXY_FPS:
            per_sec.append(np.mean(cur, axis=0))
            cur = []
    return np.array(per_sec)


def pick(stats, kind):
    n = len(stats)
    starts = np.arange(SKIP, n - SKIP - LEN)
    win = np.array([stats[s:s + LEN].mean(axis=0) for s in starts])
    tex_med = np.median(win[:, 2])
    if kind == "dark":
        ok = (win[:, 3] < 0.5) & (win[:, 2] >= tex_med)
        idx = np.where(ok)[0][np.argmin(win[ok, 0])]
    elif kind == "static":
        ok = (win[:, 2] >= tex_med) & (win[:, 0] >= 40)
        idx = np.where(ok)[0][np.argmin(win[ok, 1])]
    else:
        idx = int(np.argmax(win[:, 1]))
    return int(starts[idx]), dict(zip(("luma", "motion", "texture", "black"), map(float, win[idx])))


def cut(src, dst, start=None, frames=None):
    args = ["ffmpeg", "-v", "error", "-y"]
    if start is not None:
        args += ["-ss", str(start)]
    args += ["-i", src, "-map", "0:v:0", "-an", "-sn"]
    if frames is not None:
        args += ["-frames:v", str(frames)]
    elif start is not None:
        args += ["-t", str(LEN)]
    subprocess.run(args + ["-pix_fmt", "yuv420p", "-c:v", "ffv1", "-level", "3", dst], check=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    only = set(sys.argv[1:])
    sel_path = f"{OUT}/selection.json"
    old = json.load(open(sel_path)) if os.path.exists(sel_path) else {}
    report = {}
    for name, (src, _) in XIPH.items():
        if only and name not in only:
            continue
        if not os.path.exists(src):
            print(f"skip {name}: {src} missing", file=sys.stderr)
            continue
        info = probe(src)
        cut(src, f"{OUT}/{name}.mkv", frames=round(LEN * float(info.fps)))
        report[name] = {"source": src, "start_s": 0}
    for name, (src, kind) in FILMS.items():
        if only and name not in only:
            continue
        if not os.path.exists(src):
            print(f"skip {name}: {src} missing", file=sys.stderr)
            continue
        start, score = pick(proxy_stats(src), kind)
        cut(src, f"{OUT}/{name}.mkv", start=start)
        report[name] = {"source": src, "start_s": start, "kind": kind, **{k: round(v, 3) for k, v in score.items()}}
    for name in report:
        i = probe(f"{OUT}/{name}.mkv", count_frames=True)
        report[name].update(width=i.width, height=i.height, fps=str(i.fps), frames=i.nframes)
        print(f"{name:18s} {i.width}x{i.height} {float(i.fps):6.3f} fps {i.nframes:4d} frames  "
              + " ".join(f"{k}={v}" for k, v in report[name].items() if k in ("start_s", "luma", "motion", "texture", "black")))
    with open(sel_path, "w") as f:
        json.dump({**old, **report}, f, indent=2)


if __name__ == "__main__":
    main()
