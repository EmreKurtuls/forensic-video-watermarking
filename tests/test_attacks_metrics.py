import numpy as np

from videowm import attacks
from videowm.metrics import ber, chi_square, kl_divergence, psnr, quality, rgb_histograms
from videowm.video_io import probe

from test_ss_nonblind import N, clip


def test_metric_basics():
    a = np.zeros((8, 8), np.uint8)
    assert psnr(a, a) == 99.0 and np.isclose(psnr(a, a + 1), 48.13, atol=0.01)
    assert ber([0, 1, 1, 0], [0, 1, 0, 0]) == 0.25
    h = np.array([[10, 0, 5]])
    assert chi_square(h, h) == 0 and abs(kl_divergence(h, h)) < 1e-9
    assert chi_square(h, np.array([[0, 10, 5]])) > 0.5


def test_quality_of_identical_files(clip):
    q = quality(clip, clip, step=10)
    assert q["psnr_y"] == 99.0 and q["psnr_rgb"] == 99.0 and q["ssim_y"] == 1.0


def test_histograms_count_every_pixel(clip):
    i = probe(clip, count_frames=True)
    h = rgb_histograms(clip)
    assert h.shape == (3, 256) and (h.sum(axis=1) == i.width * i.height * i.nframes).all()


def test_attacks_produce_decodable_files(clip, tmp_path):
    i = probe(clip, count_frames=True)
    assert probe(attacks.reencode(clip, tmp_path / "a.mp4", 30, "h265"), True).nframes == i.nframes
    s = probe(attacks.scale(clip, tmp_path / "s.mp4", 0.5))
    assert (s.width, s.height) == (i.width // 2, i.height // 2)
    p, keep = attacks.trim_and_drop(clip, tmp_path / "t.mp4", i.nframes, start=5, drop_every=4)
    assert abs(probe(p, True).nframes - len(keep)) <= 1
    d, n, lost = attacks.packet_loss(clip, tmp_path / "l.mkv", rate=0.05, burst=5, seed=1)
    assert 0 < lost < n and probe(d, True).nframes > 0
