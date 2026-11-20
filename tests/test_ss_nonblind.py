import subprocess
from fractions import Fraction

import numpy as np
import pytest

from videowm.align import align, thumbnail
from videowm.dct import block_dct
from videowm.embed.ss_nonblind import DEFAULT_ALPHA, GroupMap, bit_index, embed_video, group_length, masked_alpha
from videowm.extract import identify
from videowm.keys import derive_film_keys, generate_master_key
from videowm.payload import Algorithm, PayloadCodec
from videowm.video_io import Frame, VideoWriter

W, H, N, FPS = 160, 128, 72, Fraction(24)


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("clip") / "src.mkv"
    rng = np.random.default_rng(5)
    tex = [rng.integers(0, 256, (H + 64, W + 64)).astype(np.float64) for _ in range(2)]
    tex = [np.real(np.fft.ifft2(np.fft.fft2(t) * np.exp(-0.002 * (np.add.outer(np.fft.fftfreq(H + 64) ** 2,
           np.fft.fftfreq(W + 64) ** 2) * 1e5)))) for t in tex]
    tex = [(t - t.min()) / (t.max() - t.min()) * 200 + 25 for t in tex]
    with VideoWriter(path, W, H, FPS) as wr:
        for i in range(N):
            t = tex[0] if i < 40 else tex[1]
            y = t[i % 32:i % 32 + H, (2 * i) % 64:(2 * i) % 64 + W]
            y = np.clip(y + rng.normal(0, 2, y.shape), 0, 255).astype(np.uint8)
            wr.write(Frame(y, np.full((H // 2, W // 2), 120, np.uint8), np.full((H // 2, W // 2), 130, np.uint8)))
    return path


@pytest.fixture(scope="module")
def setup(clip, tmp_path_factory):
    keys = derive_film_keys(generate_master_key(), b"TEST-0001")
    codec = PayloadCodec(keys, Algorithm.PRESENT128)
    cw = codec.encode(1001, 7)
    marked = tmp_path_factory.mktemp("marked") / "marked.mkv"
    stats = embed_video(clip, marked, cw, keys.stego_key)
    return keys, codec, cw, marked, stats


def reencode(src, dst, crf, vf=None):
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src)] + (["-vf", vf] if vf else []) + \
          ["-fps_mode", "passthrough", "-c:v", "libx264", "-crf", str(crf), str(dst)]
    subprocess.run(cmd, check=True)
    return dst


def test_group_map_keyed_jittered_lengths():
    k1, k2 = b"\x01" * 32, b"\x02" * 32
    gm = GroupMap(k1, 24)
    groups = np.array([gm.group(i) for i in range(24 * 2000)])
    assert groups[0] == 0 and np.all(np.diff(groups) >= 0) and np.all(np.diff(groups) <= 1)
    lengths = np.bincount(groups)[:-1]
    assert lengths.min() == 18 and lengths.max() == 30
    assert abs(lengths.mean() - 24) < 0.5
    counts = np.bincount(lengths - 18, minlength=13)
    assert counts.min() > 0.6 * counts.mean() and counts.max() < 1.4 * counts.mean()
    assert [GroupMap(k1, 24).group(i) for i in range(500)] == list(groups[:500])
    assert [GroupMap(k2, 24).group(i) for i in range(500)] != list(groups[:500])
    assert gm.group(10 ** 6) > gm.group(10 ** 6 - 100)
    assert GroupMap(k1, 24).complete_groups(0) == 0
    starts = np.flatnonzero(np.diff(groups)) + 1
    assert np.mean(starts % 24 == 0) < 0.2


def test_helpers():
    assert group_length(Fraction(30000, 1001)) == 30 and group_length(24) == 24 and group_length(25) == 25
    assert list(bit_index(0, 5, 188)) == [0, 1, 2, 3, 4]
    assert bit_index(1, 100, 188)[0] == 100 and bit_index(1, 100, 188)[-1] == 11
    d = block_dct(np.random.default_rng(0).integers(0, 256, (64, 64)).astype(np.uint8))
    a = masked_alpha(d, 3.0)
    assert np.isclose(np.sqrt(np.mean(a ** 2)), 3.0)


def test_quality(setup):
    stats = setup[4]
    assert stats.frames == N and stats.group_len == 24
    assert 45 < stats.psnr_y < 55


def test_lossless_extraction_is_error_free(setup, clip):
    keys, codec, cw, marked, _ = setup
    ident, ex = identify(marked, clip, codec)
    assert (ident.user_id, ident.session) == (1001, 7)
    assert np.array_equal(ex.bits, cw) and ex.match == list(range(N))


def test_survives_reencode(setup, clip, tmp_path):
    keys, codec, cw, marked, _ = setup
    ident, _ = identify(reencode(marked, tmp_path / "p.mp4", 23), clip, codec)
    assert ident is not None and (ident.user_id, ident.session) == (1001, 7)


def test_survives_trim_and_frame_drop_with_alignment(setup, clip, tmp_path):
    keys, codec, cw, marked, _ = setup
    keep = [i for i in range(6, N) if i % 7 != 0]
    sel = "+".join(f"eq(n\\,{i})" for i in keep)
    pir = reencode(marked, tmp_path / "p.mp4", 23, f"select='{sel}',setpts=N/24/TB")
    ident, ex = identify(pir, clip, codec)
    assert ident is not None and (ident.user_id, ident.session) == (1001, 7)
    assert len(ex.match) >= len(keep) - 1
    assert np.mean(np.array(ex.match) // 24 == np.array(keep[:len(ex.match)]) // 24) > 0.9


def test_unmarked_or_wrong_key_gives_no_identity(setup, clip):
    keys, codec, cw, marked, _ = setup
    assert identify(clip, clip, codec)[0] is None
    other = PayloadCodec(derive_film_keys(generate_master_key(), b"TEST-0001"), Algorithm.PRESENT128)
    assert identify(marked, clip, other)[0] is None


def test_align_handles_trim_and_drops():
    rng = np.random.default_rng(2)
    orig = [rng.normal(128, 40, (32, 32)) for _ in range(50)]
    keep = [i for i in range(5, 50) if i % 4 != 0]
    pir = [orig[i] + rng.normal(0, 3, (32, 32)) for i in keep]
    assert align(pir, orig) == keep


def test_thumbnail_shape():
    assert thumbnail(np.zeros((288, 352), np.uint8)).shape == (32, 32)
    assert thumbnail(np.zeros((180, 320), np.uint8)).shape == (32, 32)


def test_frames_to_identify(setup, clip):
    from videowm.extract import frames_to_identify
    keys, codec, cw, marked, _ = setup
    ident, ex = identify(marked, clip, codec, keep_per_frame=True)
    assert ex.per_frame.shape == (N, 188) and np.allclose(ex.per_frame.sum(axis=0), ex.soft)
    n = frames_to_identify(ex.per_frame, codec, (1001, 7))
    assert n is not None and 1 <= n <= N
    assert frames_to_identify(ex.per_frame, codec, (1001, 8)) is None


def test_align_identical_black_start_and_near_duplicates():
    rng = np.random.default_rng(4)
    content = [rng.normal(128, 40, (32, 32)) for _ in range(200)]
    content[120] = content[121] + 0.01
    black = [np.zeros((32, 32)) for _ in range(30)]
    orig = black + content + black + content
    pir = [f + rng.normal(0, 2, (32, 32)) for f in orig]
    assert align(pir, orig) == list(range(len(orig)))
    keep = [i for i in range(10, len(orig)) if i % 9 != 0]
    m = align([orig[i] + rng.normal(0, 2, (32, 32)) for i in keep], orig)
    is_black = lambda j: not orig[j].any()
    for got, want in zip(m, keep):
        assert got == want or (is_black(got) and is_black(want))
    assert all(g == w for g, w in zip(m, keep) if not is_black(w))
