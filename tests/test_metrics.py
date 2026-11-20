import subprocess
from fractions import Fraction

import numpy as np
import pytest

from videowm.metrics import vmaf
from videowm.video_io import Frame, VideoWriter

pytestmark = pytest.mark.skipif(
    "libvmaf" not in subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout,
    reason="FFmpeg without libvmaf")


@pytest.fixture(scope="module")
def moving(tmp_path_factory):
    d = tmp_path_factory.mktemp("vmaf")
    src = d / "src.mkv"
    rng = np.random.default_rng(1)
    tex = rng.integers(16, 236, (288 + 400, 352)).astype(np.uint8)
    with VideoWriter(src, 352, 288, Fraction(24)) as wr:
        for i in range(48):
            y = tex[8 * i:8 * i + 288]
            wr.write(Frame(y.copy(), np.full((144, 176), 128, np.uint8), np.full((144, 176), 128, np.uint8)))
    mp4 = d / "crf10.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:v", "libx264", "-crf", "10", "-bf", "3",
                    "-pix_fmt", "yuv420p", str(mp4)], check=True)
    return src, mp4


def test_vmaf_identical_is_near_100(moving):
    src, _ = moving
    assert vmaf(src, src, threads=1) > 99


def test_vmaf_pairs_frames_by_index(moving):
    src, mp4 = moving
    assert vmaf(src, mp4, threads=1) > 90
