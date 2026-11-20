import os
import subprocess

import numpy as np
import pytest

from videowm.embed import lsb, qim_blind
from videowm.keys import derive_film_keys, generate_master_key
from videowm.payload import Algorithm, NoValidMark, PayloadCodec
from videowm.video_io import probe, read_frames

from test_ss_nonblind import N, clip


@pytest.fixture(scope="module")
def keys():
    return derive_film_keys(generate_master_key(), b"TEST-0001")


def test_qim_lossless_blind_roundtrip(clip, keys, tmp_path):
    codec = PayloadCodec(keys, Algorithm.FF1)
    cw = codec.encode(42, 9)
    m = tmp_path / "q.mkv"
    st = qim_blind.embed_video(clip, m, cw, keys.stego_key, delta=24)
    assert st.frames == N and st.psnr_y > 40
    soft = qim_blind.extract_soft(m, keys.stego_key, 24, 188)
    ident = codec.decode((soft > 0).astype(np.uint8))
    assert (ident.user_id, ident.session) == (42, 9)


def test_qim_wrong_stego_key_gives_no_identity(clip, keys, tmp_path):
    codec = PayloadCodec(keys, Algorithm.PRESENT128)
    m = tmp_path / "q.mkv"
    qim_blind.embed_video(clip, m, codec.encode(42, 9), keys.stego_key, delta=24)
    soft = qim_blind.extract_soft(m, os.urandom(32), 24, 188)
    with pytest.raises(NoValidMark):
        codec.decode((soft > 0).astype(np.uint8))


def test_lsb_large_payload_lossless(clip, keys, tmp_path):
    info = probe(clip, count_frames=True)
    cap = lsb.capacity_bits(info.width, info.height, info.nframes) // 8
    data = os.urandom(cap - 100)
    for ext in (".mp4", ".mkv", ".avi"):
        m = tmp_path / f"l{ext}"
        st = lsb.embed_video(clip, m, data, keys.stego_key)
        assert st.psnr_y > 50
        assert lsb.extract_bytes(m, len(data)) == data


def test_lsb_matching_changes_pixels_by_at_most_one(clip, keys, tmp_path):
    m = tmp_path / "l.mkv"
    lsb.embed_video(clip, m, os.urandom(5000), keys.stego_key)
    for a, b in zip(read_frames(clip), read_frames(m)):
        assert np.max(np.abs(a.y.astype(int) - b.y.astype(int))) <= 1
        assert np.array_equal(a.u, b.u) and np.array_equal(a.v, b.v)


def test_lsb_rejects_payload_over_capacity(clip, keys, tmp_path):
    info = probe(clip, count_frames=True)
    with pytest.raises(ValueError):
        lsb.embed_video(clip, tmp_path / "l.mkv", bytes(lsb.capacity_bits(info.width, info.height, info.nframes) // 8 + 1),
                        keys.stego_key)


def test_lsb_dies_on_lossy_reencode(clip, keys, tmp_path):
    data = os.urandom(2000)
    m = tmp_path / "l.mkv"
    lsb.embed_video(clip, m, data, keys.stego_key)
    p = tmp_path / "p.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(m), "-c:v", "libx264", "-crf", "18", str(p)], check=True)
    bits = lsb.extract_bits(p, 8 * len(data))
    ber = np.mean(bits != np.unpackbits(np.frombuffer(data, np.uint8)))
    assert ber > 0.3
