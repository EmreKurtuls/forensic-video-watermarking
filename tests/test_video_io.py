import subprocess
from fractions import Fraction

import numpy as np
import pytest

from videowm.video_io import Frame, VideoWriter, probe, read_frames

W, H, N = 64, 48, 12
FPS = Fraction(24000, 1001)


def synthetic_frames():
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:H, 0:W]
    for t in range(N):
        y = ((xx * 3 + yy * 2 + t * 5) % 256).astype(np.uint8) ^ rng.integers(0, 16, (H, W), dtype=np.uint8)
        u = rng.integers(0, 256, (H // 2, W // 2), dtype=np.uint8)
        v = np.full((H // 2, W // 2), 128 + t, np.uint8)
        yield Frame(y, u, v)


@pytest.mark.parametrize("ext", [".mp4", ".mkv", ".avi"])
def test_lossless_roundtrip(tmp_path, ext):
    path = tmp_path / f"clip{ext}"
    src = list(synthetic_frames())
    with VideoWriter(path, W, H, FPS) as wr:
        for f in src:
            wr.write(f)
    info = probe(path, count_frames=True)
    assert (info.width, info.height, info.nframes) == (W, H, N)
    assert info.fps == FPS
    out = list(read_frames(path))
    assert len(out) == N
    for a, b in zip(src, out):
        assert np.array_equal(a.y, b.y) and np.array_equal(a.u, b.u) and np.array_equal(a.v, b.v)


def test_rescaled_read(tmp_path):
    path = tmp_path / "clip.mkv"
    with VideoWriter(path, W, H, FPS) as wr:
        for f in synthetic_frames():
            wr.write(f)
    out = list(read_frames(path, width=2 * W, height=2 * H))
    assert len(out) == N and out[0].y.shape == (2 * H, 2 * W) and out[0].u.shape == (H, W)


def test_early_stop_does_not_hang(tmp_path):
    path = tmp_path / "clip.mkv"
    with VideoWriter(path, W, H, FPS) as wr:
        for f in synthetic_frames():
            wr.write(f)
    gen = read_frames(path)
    next(gen)
    gen.close()


def test_writer_rejects_unknown_extension(tmp_path):
    with pytest.raises(ValueError):
        VideoWriter(tmp_path / "x.webm", W, H, FPS)


@pytest.mark.parametrize("ext", [".mp4", ".mkv", ".avi"])
def test_colour_tags_are_copied(tmp_path, ext):
    src = tmp_path / "src.mkv"
    with VideoWriter(src, 64, 48, Fraction(25),
                     color=("-vf", "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709")) as wr:
        for f in synthetic_frames():
            wr.write(f)
    info = probe(src)
    assert "colorspace=bt709" in info.color[1]
    dst = tmp_path / f"dst{ext}"
    with VideoWriter(dst, info.width, info.height, info.fps, color=info.color) as wr:
        for f in read_frames(src):
            wr.write(f)
    for a, b in zip(read_frames(src), read_frames(dst)):
        assert all(np.array_equal(p, q) for p, q in ((a.y, b.y), (a.u, b.u), (a.v, b.v)))
    if ext == ".avi":
        assert probe(dst).color == ()
    else:
        assert "colorspace=bt709" in probe(dst).color[1]


@pytest.mark.parametrize("ext", [".mp4", ".mkv"])
def test_pixel_aspect_is_copied(tmp_path, ext):
    src = tmp_path / "src.mkv"
    with VideoWriter(src, 64, 48, Fraction(25), color=("-vf", "setsar=r=128/117:max=65535")) as wr:
        for f in synthetic_frames():
            wr.write(f)
    info = probe(src)
    assert "setsar=r=128/117" in info.color[1]
    dst = tmp_path / f"dst{ext}"
    with VideoWriter(dst, info.width, info.height, info.fps, color=info.color) as wr:
        for f in read_frames(src):
            wr.write(f)
    sar = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=sample_aspect_ratio",
                          "-of", "csv=p=0", str(dst)], capture_output=True, text=True).stdout.strip()
    assert sar == "128:117"
    for a, b in zip(read_frames(src), read_frames(dst)):
        assert all(np.array_equal(p, q) for p, q in ((a.y, b.y), (a.u, b.u), (a.v, b.v)))

