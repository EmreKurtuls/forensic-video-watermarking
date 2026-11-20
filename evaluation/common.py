import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from videowm.embed import lsb, qim_blind, ss_nonblind
from videowm.extract import identify
from videowm.keys import derive_film_keys, generate_master_key
from videowm.payload import Algorithm, NoValidMark, PayloadCodec

CLIPS_DIR = ROOT / "data" / "clips"
WORK = ROOT / "data" / "out" / "eval"
RESULTS = Path(os.environ.get("VIDEOWM_RESULTS", ROOT / "results"))
WORK.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(exist_ok=True)

CLIPS = ["akiyo_cif", "foreman_cif", "park_joy_720p", "bbb_action_720p", "sintel_dark_720p", "tos_static_720p"]
CLIP_1080 = "crowd_run_1080p"
if os.environ.get("VIDEOWM_CLIPS"):
    CLIPS, CLIP_1080 = os.environ["VIDEOWM_CLIPS"].split(","), os.environ["VIDEOWM_CLIPS"].split(",")[0]
FORMATS = [".mp4", ".mkv", ".avi"]
ALGS = {"PRESENT-128": Algorithm.PRESENT128, "FF1": Algorithm.FF1}
ALPHA = ss_nonblind.DEFAULT_ALPHA
DELTAS = {"QIM D=24": qim_blind.DELTA_EQUAL_QUALITY, "QIM D=64": qim_blind.DELTA_EQUAL_ROBUSTNESS}
USER, SESSION = 1001, 7


def clip_path(name: str) -> Path:
    return CLIPS_DIR / f"{name}.mkv"


def available(names):
    out = [n for n in names if clip_path(n).exists()]
    for n in set(names) - set(out):
        print(f"[skip] {n}: clip missing", file=sys.stderr)
    return out


def film_setup(name: str, alg=Algorithm.PRESENT128, master: bytes | None = None):
    keys = derive_film_keys(master or generate_master_key(), name.encode())
    return keys, PayloadCodec(keys, alg)


def decode_or_none(codec, bits):
    try:
        return codec.decode(np.asarray(bits, np.uint8))
    except NoValidMark:
        return None


class Timer:
    def __enter__(self):
        self.t0, self.c0 = time.perf_counter(), time.process_time()
        return self

    def __exit__(self, *exc):
        self.wall, self.cpu = time.perf_counter() - self.t0, time.process_time() - self.c0


class CSV:

    def __init__(self, name: str, fields: list[str]):
        self.path = RESULTS / f"{name}.csv"
        self.f = open(self.path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fields)
        self.w.writeheader()

    def row(self, **kw):
        self.w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in kw.items()})
        self.f.flush()
        print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in kw.items()), flush=True)

    def close(self):
        self.f.close()


def embed_ss(src, dst, cw, keys, alpha=ALPHA):
    return ss_nonblind.embed_video(src, dst, cw, keys.stego_key, alpha)


def extract_ss(pirate, src, codec, aligned=True, keep_per_frame=False):
    ident, ex = identify(pirate, src, codec, aligned=aligned, keep_per_frame=keep_per_frame)
    return ident, ex.bits, ex


def embed_qim(src, dst, cw, keys, delta):
    return qim_blind.embed_video(src, dst, cw, keys.stego_key, delta)


def extract_qim(pirate, src_info, codec, delta):
    soft = qim_blind.extract_soft(pirate, codec.keys.stego_key, delta, len(codec.encode(0, 0)),
                                  fps=src_info.fps, width=src_info.width, height=src_info.height)
    bits = qim_blind.decide(soft)
    return decode_or_none(codec, bits), bits


def lsb_payload(codec, frame_bytes: int) -> bytes:
    cw = codec.encode_bytes(USER, SESSION)
    return (cw * (frame_bytes // len(cw)))


def extract_lsb(pirate, codec, frame_bytes: int):
    reps = frame_bytes // 24
    bits = lsb.extract_bits(pirate, reps * 24 * 8).reshape(reps, 192)[:, :188]
    maj = (bits.mean(axis=0) > 0.5).astype(np.uint8)
    return decode_or_none(codec, maj), maj
