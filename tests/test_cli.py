import os
import stat
import subprocess

import pytest

from videowm.cli import main
from videowm.db import PlatformDB

from test_ss_nonblind import clip


@pytest.fixture
def platform(tmp_path, clip):
    key, db = tmp_path / "platform.key", tmp_path / "platform.sqlite"
    assert main(["keygen", "--key", str(key)]) == 0
    return key, db, clip, tmp_path


def test_keygen_file_is_private_and_not_overwritten(platform):
    key = platform[0]
    assert stat.S_IMODE(os.stat(key).st_mode) == 0o600 and os.path.getsize(key) == 32
    before = key.read_bytes()
    assert main(["keygen", "--key", str(key)]) == 1
    assert key.read_bytes() == before


@pytest.mark.parametrize("alg", ["present", "ff1"])
def test_end_to_end_trace(platform, alg, capsys):
    key, db, clip, d = platform
    base = ["--db", str(db), "--key", str(key), "--film-id", "TEST-0001"]
    assert main(["register-film", "--db", str(db), "--film-id", "TEST-0001", "--original", str(clip), "--alg", alg]) == 0
    for uid in (1001, 2002):
        assert main(["embed", *base, "--user-id", str(uid), "--out", str(d / f"u{uid}.mkv")]) == 0
    assert main(["embed", *base, "--user-id", "2002", "--out", str(d / "u2002b.mkv")]) == 0
    pirate = d / "leak.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(d / "u2002b.mkv"), "-c:v", "libx264", "-crf", "23",
                    str(pirate)], check=True)
    capsys.readouterr()
    assert main(["trace", *base, "--pirate", str(pirate)]) == 0
    out = capsys.readouterr().out
    assert "user 2002, session 1" in out and "u2002b.mkv" in out


def test_trace_unmarked_and_wrong_key(platform, capsys):
    key, db, clip, d = platform
    base = ["--db", str(db), "--film-id", "TEST-0001"]
    main(["register-film", *base, "--original", str(clip)])
    main(["embed", *base, "--key", str(key), "--user-id", "7", "--out", str(d / "u7.mkv")])
    assert main(["trace", *base, "--key", str(key), "--pirate", str(clip)]) == 2
    other = d / "other.key"
    main(["keygen", "--key", str(other)])
    assert main(["trace", *base, "--key", str(other), "--pirate", str(d / "u7.mkv")]) == 2
    assert "no valid watermark found" in capsys.readouterr().out


def test_sessions_are_unique_and_counted_per_user(tmp_path):
    db = PlatformDB(tmp_path / "p.sqlite")
    assert [db.next_session(5) for _ in range(3)] == [0, 1, 2]
    assert db.next_session(6) == 0
    db.close()
    assert PlatformDB(tmp_path / "p.sqlite").next_session(5) == 3


def test_unknown_film_and_duplicate_registration(platform, capsys):
    key, db, clip, d = platform
    assert main(["trace", "--db", str(db), "--key", str(key), "--film-id", "NOPE", "--pirate", str(clip)]) == 1
    args = ["register-film", "--db", str(db), "--film-id", "F", "--original", str(clip)]
    assert main(args) == 0 and main(args) == 1
