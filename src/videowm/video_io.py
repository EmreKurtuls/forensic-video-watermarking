import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

LOSSLESS_CODECS = {
    ".mp4": ["-c:v", "libx264", "-qp", "0", "-preset", "medium"],
    ".mkv": ["-c:v", "ffv1", "-level", "3"],
    ".avi": ["-c:v", "ffv1", "-level", "3"],
}


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: Fraction
    nframes: int | None
    codec: str
    pix_fmt: str
    color: tuple = ()


@dataclass
class Frame:
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray


def probe(path, count_frames: bool = False) -> VideoInfo:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,codec_name,pix_fmt,"
           "color_range,color_space,color_primaries,color_transfer,sample_aspect_ratio",
           "-of", "json"]
    if count_frames:
        cmd.append("-count_frames")
    s = json.loads(subprocess.run(cmd + [str(path)], capture_output=True, check=True, text=True).stdout)["streams"][0]
    rate = s.get("avg_frame_rate", "0/0")
    if rate in ("0/0", "0"):
        rate = s["r_frame_rate"]
    n = s.get("nb_read_frames") if count_frames else s.get("nb_frames")
    return VideoInfo(int(s["width"]), int(s["height"]), Fraction(rate),
                     int(n) if n not in (None, "N/A") else None, s["codec_name"], s.get("pix_fmt", ""),
                     color_args(s))


_COLOR_TAGS = {"color_range": "range", "color_space": "colorspace",
               "color_primaries": "color_primaries", "color_transfer": "color_trc"}


def color_args(stream: dict) -> tuple:
    tags = [f"{name}={stream[key]}" for key, name in _COLOR_TAGS.items()
            if stream.get(key) and stream[key] not in ("unknown", "reserved", "unspecified")]
    filters = ["setparams=" + ":".join(tags)] if tags else []
    sar = stream.get("sample_aspect_ratio", "")
    if ":" in sar and sar != "0:1":
        filters.append(f"setsar=r={sar.replace(':', '/')}:max=65535")
    return ("-vf", ",".join(filters)) if filters else ()


def read_frames(path, width: int | None = None, height: int | None = None, vf: str | None = None,
                passthrough: bool = True):
    if width is None or height is None:
        info = probe(path)
        width, height = info.width, info.height
    if width % 2 or height % 2:
        raise ValueError("YUV 4:2:0 needs even width and height")
    filters = [vf] if vf else []
    filters.append(f"scale={width}:{height}:flags=bicubic")
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:v:0", "-vf", ",".join(filters)]
    if passthrough:
        cmd += ["-fps_mode", "passthrough"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]
    ysz, csz = width * height, (width // 2) * (height // 2)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            buf = proc.stdout.read(ysz + 2 * csz)
            if len(buf) < ysz + 2 * csz:
                break
            a = np.frombuffer(buf, np.uint8)
            yield Frame(a[:ysz].reshape(height, width).copy(),
                        a[ysz:ysz + csz].reshape(height // 2, width // 2).copy(),
                        a[ysz + csz:].reshape(height // 2, width // 2).copy())
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()


class VideoWriter:

    def __init__(self, path, width: int, height: int, fps: Fraction, codec_args: list[str] | None = None,
                 color: tuple = ()):
        path = Path(path)
        if codec_args is None:
            if path.suffix.lower() not in LOSSLESS_CODECS:
                raise ValueError(f"no lossless preset for {path.suffix}; pass codec_args")
            codec_args = LOSSLESS_CODECS[path.suffix.lower()]
        self.width, self.height, self.path = width, height, path
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y",
               "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{width}x{height}", "-r", str(Fraction(fps)), "-i", "-",
               *codec_args, *color, "-pix_fmt", "yuv420p", str(path)]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: Frame) -> None:
        if frame.y.shape != (self.height, self.width):
            raise ValueError("frame size does not match the writer")
        for plane in (frame.y, frame.u, frame.v):
            self._proc.stdin.write(np.ascontiguousarray(plane, dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self._proc.stdin and not self._proc.stdin.closed:
            self._proc.stdin.close()
        err = self._proc.stderr.read().decode(errors="replace")
        if self._proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path}: {err.strip()}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
