import subprocess

import numpy as np

TS_PACKET = 188


def _ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *map(str, args)], check=True)


def reencode(src, dst, crf: int, codec: str = "h264", preset: str = "medium", vf: str | None = None):
    lib = {"h264": "libx264", "h265": "libx265"}[codec]
    extra = ["-x265-params", "log-level=error"] if codec == "h265" else []
    _ff("-i", src, *(["-vf", vf] if vf else []), "-fps_mode", "passthrough", "-an",
        "-c:v", lib, "-crf", crf, "-preset", preset, *extra, "-pix_fmt", "yuv420p", dst)
    return dst


def scale(src, dst, factor: float, crf: int = 23):
    vf = f"scale=trunc(iw*{factor}/2)*2:trunc(ih*{factor}/2)*2:flags=bicubic"
    return reencode(src, dst, crf, vf=vf)


def trim_and_drop(src, dst, total_frames: int, start: int, drop_every: int, crf: int = 23):
    keep = [i for i in range(start, total_frames) if drop_every <= 0 or i % drop_every != 0]
    sel = "+".join(f"eq(n\\,{i})" for i in keep)
    reencode(src, dst, crf, vf=f"select='{sel}',setpts=N/FRAME_RATE/TB")
    return dst, keep


def packet_loss(src, dst, rate: float, burst: int = 1, crf: int = 23, seed: int = 0):
    ts = str(dst) + ".ts"
    from .video_io import probe
    gop = 2 * round(float(probe(src).fps))
    _ff("-i", src, "-an", "-c:v", "libx264", "-crf", crf, "-preset", "medium",
        "-x264-params", f"keyint={gop}:min-keyint={gop}:repeat-headers=1", "-f", "mpegts", ts)
    data = open(ts, "rb").read()
    n = len(data) // TS_PACKET
    rng = np.random.default_rng(seed)
    lost = np.zeros(n, bool)
    events = rng.random(n) < rate / burst
    for i in np.flatnonzero(events):
        lost[i:i + burst] = True
    keep = np.repeat(~lost, TS_PACKET)
    body = np.frombuffer(data[:n * TS_PACKET], np.uint8)[keep].tobytes()
    with open(ts, "wb") as f:
        f.write(body)
    import os
    if os.path.exists(dst):
        os.remove(dst)
    r = subprocess.run(["ffmpeg", "-v", "quiet", "-y", "-err_detect", "ignore_err", "-i", ts, "-an",
                        "-fps_mode", "passthrough", "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(dst)], check=False)
    try:
        probe(dst)
    except Exception:
        if os.path.exists(dst):
            os.remove(dst)
    return dst, n, int(lost.sum())
