import re
import subprocess

import numpy as np
from skimage.metrics import structural_similarity

from .video_io import probe, read_frames


def psnr(a: np.ndarray, b: np.ndarray, peak: float = 255.0) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else float(10 * np.log10(peak ** 2 / mse))


def ber(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.mean(a != b))


def read_rgb(path, width=None, height=None):
    if width is None:
        i = probe(path)
        width, height = i.width, i.height
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = width * height * 3
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()


def quality(original, marked, step: int = 1) -> dict:
    py, sy, prgb = [], [], []
    for i, (a, b) in enumerate(zip(read_frames(original), read_frames(marked))):
        if i % step:
            continue
        py.append(psnr(a.y, b.y))
        sy.append(structural_similarity(a.y, b.y, data_range=255))
    for i, (a, b) in enumerate(zip(read_rgb(original), read_rgb(marked))):
        if i % step == 0:
            prgb.append(psnr(a, b))
    return {"psnr_y": float(np.mean(py)), "psnr_rgb": float(np.mean(prgb)),
            "ssim_y": float(np.mean(sy)), "ssim_y_min": float(np.min(sy))}


def rgb_histograms(path, step: int = 1) -> np.ndarray:
    h = np.zeros((3, 256), np.int64)
    for i, f in enumerate(read_rgb(path)):
        if i % step == 0:
            for c in range(3):
                h[c] += np.bincount(f[..., c].ravel(), minlength=256)
    return h


def chi_square(h_ref: np.ndarray, h_test: np.ndarray) -> float:
    p, q = h_ref / h_ref.sum(), h_test / h_test.sum()
    s = p + q
    m = s > 0
    return float(np.sum((p[m] - q[m]) ** 2 / s[m]))


def kl_divergence(h_ref: np.ndarray, h_test: np.ndarray, eps: float = 1e-10) -> float:
    p = h_ref / h_ref.sum() + eps
    q = h_test / h_test.sum() + eps
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log2(p / q)))


def vmaf(original, distorted, threads: int = 4) -> float:
    graph = ("[0:v]settb=1/1000,setpts=N*40[d];[1:v]settb=1/1000,setpts=N*40[r];"
             f"[d][r]libvmaf=n_threads={threads}:shortest=1")
    out = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-i", str(distorted), "-i", str(original),
                          "-lavfi", graph, "-f", "null", "-"], capture_output=True, text=True)
    found = re.findall(r"VMAF score: ([0-9.]+)", out.stderr)
    if not found:
        raise RuntimeError("libvmaf failed (is FFmpeg built with --enable-libvmaf?): " + out.stderr[-300:])
    return float(found[-1])

