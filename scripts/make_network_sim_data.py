import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from videowm import attacks
from videowm.db import PlatformDB
from videowm.embed import ss_nonblind
from videowm.embed.ss_nonblind import DEFAULT_ALPHA, GroupMap, bit_index, embed_luma, group_length
from videowm.extract import identify
from videowm.keys import derive_film_keys, generate_master_key, pattern
from videowm.dct import LOW_BAND
from videowm.payload import Algorithm, NoValidMark, PayloadCodec, pack_block
from videowm.video_io import probe, read_frames

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation" / "network_sim"
WORK = ROOT / "data" / "out" / "network_sim"
SRC = ROOT / "data" / "clips" / "bbb_action_720p.mkv"
FILM_ID = "film-0108"
VIEWERS = [(1001, "TV"), (1002, "Laptop"), (1003, "Phone")]
LEAKER = 1002
ATTACK = (0.5, 33)
FRAME = 36
ALGS = [("PRESENT-128", Algorithm.PRESENT128), ("FF1", Algorithm.FF1)]


def jpg_from_video(path, idx, name, width=640):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-vf", f"select=eq(n\\,{idx}),scale={width}:-2",
                    "-frames:v", "1", "-q:v", "3", str(OUT / name)], check=True)


def jpg_from_gray(y, name, width=640):
    h, w = y.shape
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-i", "-",
                    "-vf", f"scale={width}:-2", "-q:v", "3", str(OUT / name)], input=y.tobytes(), check=True)


def time_us(fn, n=30):
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return round(float(np.median(t)) * 1e6, 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    info = probe(SRC)
    master = generate_master_key()
    keys = derive_film_keys(master, FILM_ID.encode())
    with tempfile.TemporaryDirectory() as td:
        db = PlatformDB(Path(td) / "platform.db")
        for _ in range(6):
            db.next_session(1001)
        for _ in range(2):
            db.next_session(1002)
        sessions = {u: db.next_session(u) for u, _ in VIEWERS}
        db.close()

    fr = next(f for i, f in enumerate(read_frames(SRC)) if i == FRAME)
    G = group_length(info.fps)
    g = GroupMap(keys.stego_key, info.fps).group(FRAME)
    nb = (info.height // 8) * (info.width // 8)
    u = pattern(keys.stego_key, g, nb, len(LOW_BAND))
    jpg_from_video(SRC, FRAME, "frame.jpg")

    data = {"film": {"id": FILM_ID, "width": info.width, "height": info.height,
                     "fps": float(info.fps), "seconds": round(info.nframes / float(info.fps), 1) if info.nframes else 10,
                     "blocks": nb, "group": G, "alpha": round(DEFAULT_ALPHA, 2)},
            "viewers": [{"id": uid, "device": dev, "session": sessions[uid]} for uid, dev in VIEWERS],
            "leaker": LEAKER, "attack": {"scale": ATTACK[0], "crf": ATTACK[1]}, "algs": {}}

    for label, alg in ALGS:
        codec = PayloadCodec(keys, alg)
        rows = []
        for uid, _ in VIEWERS:
            s = sessions[uid]
            block = pack_block(uid, s)
            cw_bytes = codec.encode_bytes(uid, s)
            cw = codec.encode(uid, s)
            rows.append({"id": uid, "block": f"{block:016x}", "cipher": cw_bytes[:8].hex(), "tag": cw_bytes[8:12].hex(),
                         "codeword": "".join(map(str, cw)), "encode_us": time_us(lambda: codec.encode_bytes(uid, s))})
            if label == "PRESENT-128":
                signs = cw[bit_index(g, nb, len(cw))] * 2.0 - 1.0
                y2 = embed_luma(fr.y, signs, u)
                diff = np.clip(128 + 20 * (y2.astype(np.int16) - fr.y), 0, 255).astype(np.uint8)
                jpg_from_gray(diff, f"diff_{uid}.jpg")
        enc_us = time_us(lambda: codec._enc(pack_block(LEAKER, sessions[LEAKER])), 200)

        cw = codec.encode(LEAKER, sessions[LEAKER])
        marked = WORK / f"copy_{LEAKER}_{label}.mkv"
        t0 = time.perf_counter()
        ss_nonblind.embed_video(SRC, marked, cw, keys.stego_key)
        embed_s = time.perf_counter() - t0
        pirate = attacks.scale(marked, WORK / f"pirate_{label}.mp4", ATTACK[0], crf=ATTACK[1])
        t0 = time.perf_counter()
        ident, ex = identify(pirate, SRC, codec)
        trace_s = time.perf_counter() - t0
        got = ex.bits
        errors = [int(i) for i in np.flatnonzero(got != cw)]
        scores = {}
        for r in rows:
            c = np.array([int(b) for b in r["codeword"]]) * 2.0 - 1.0
            scores[r["id"]] = round(float(np.sum(ex.soft * c) / np.sum(np.abs(ex.soft))), 3)
        wrong = PayloadCodec(derive_film_keys(generate_master_key(), FILM_ID.encode()), alg)
        try:
            wrong.decode(got)
            wrong_result = "identity"
        except NoValidMark:
            wrong_result = "NoValidMark"
        pi = probe(pirate, count_frames=True)
        if label == "PRESENT-128":
            jpg_from_video(pirate, FRAME, "pirate.jpg")
        data["algs"][label] = {
            "rows": rows, "encrypt_us": enc_us, "embed_s": round(embed_s, 1), "trace_s": round(trace_s, 1),
            "trace": {"pirate_w": pi.width, "pirate_h": pi.height, "pirate_frames": pi.nframes,
                      "matched": int(len(ex.match)), "extracted": "".join(map(str, got)), "errors": errors,
                      "corrected": None if ident is None else ident.corrected_bits,
                      "user": None if ident is None else ident.user_id,
                      "session": None if ident is None else ident.session,
                      "scores": scores, "wrong_key": wrong_result}}
        print(label, "errors", len(errors), "->", ident, "scores", scores, "wrong key:", wrong_result, flush=True)
        marked.unlink()

    (OUT / "data.js").write_text("window.SIM = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print("wrote", OUT / "data.js")


if __name__ == "__main__":
    main()
