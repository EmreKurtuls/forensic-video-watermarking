import hmac
import os
import time

import numpy as np

from common import (CLIPS, CSV, SESSION, USER, WORK, available, clip_path, decode_or_none, embed_ss, extract_ss,
                    film_setup)
from videowm.keys import FilmKeys, derive_film_keys, generate_master_key
from videowm.payload import Algorithm, NoValidMark, PayloadCodec


def with_keys(keys: FilmKeys, **kw) -> FilmKeys:
    d = dict(film_id=keys.film_id, enc_key=keys.enc_key, mac_key=keys.mac_key, stego_key=keys.stego_key)
    d.update(kw)
    return FilmKeys(**d)


def video_level():
    out = CSV("c_video", ["clip", "algorithm", "case", "identity_returned", "user_id", "raw_bit_errors"])
    for clip in available(CLIPS):
        src = clip_path(clip)
        for alg in (Algorithm.PRESENT128, Algorithm.FF1):
            master = generate_master_key()
            keys, codec = film_setup(clip, alg, master)
            cw = codec.encode(USER, SESSION)
            marked = WORK / f"c_{clip}.mkv"
            embed_ss(src, marked, cw, keys)
            other = derive_film_keys(master, b"OTHER-FILM")
            cases = {
                "correct keys (control)": (marked, keys),
                "wrong master key": (marked, derive_film_keys(generate_master_key(), clip.encode())),
                "wrong enc_key only": (marked, with_keys(keys, enc_key=os.urandom(16))),
                "wrong mac_key only": (marked, with_keys(keys, mac_key=os.urandom(16))),
                "wrong stego_key only": (marked, with_keys(keys, stego_key=os.urandom(32))),
                "wrong film_id": (marked, derive_film_keys(master, b"WRONG-FILM")),
                "unmarked original": (src, keys),
            }
            replay = WORK / f"c_{clip}_replay.mkv"
            embed_ss(src, replay, PayloadCodec(other, alg).encode(USER, SESSION), keys)
            cases["cross-film replay"] = (replay, keys)
            frame = WORK / f"c_{clip}_forged.mkv"
            fb = np.unpackbits(np.frombuffer(os.urandom(4), np.uint8))
            payload = np.packbits(np.concatenate([cw[:64], fb])).tobytes()
            embed_ss(src, frame, np.unpackbits(np.frombuffer(payload + codec._bch.encode(payload), np.uint8))[:188], keys)
            cases["framing: forged tag"] = (frame, keys)
            for case, (video, k) in cases.items():
                ident, bits, _ = extract_ss(video, src, PayloadCodec(k, alg))
                out.row(clip=clip, algorithm=alg.name, case=case, identity_returned=ident is not None,
                        user_id=ident.user_id if ident else "", raw_bit_errors=int(np.sum(bits != cw)))
            for f in (marked, replay, frame):
                os.remove(f)
    out.close()


def bit_level(trials=100_000, enc_trials=20_000):
    out = CSV("c_bits", ["test", "trials", "identities_returned", "note"])
    keys, codec = film_setup("BITS")
    rng = np.random.default_rng(1)
    acc = sum(decode_or_none(codec, rng.integers(0, 2, 188, dtype=np.uint8)) is not None for _ in range(trials))
    out.row(test="random 188-bit words", trials=trials, identities_returned=acc, note="expected ~0")
    cw = codec.encode(USER, SESSION)
    for alg in (Algorithm.PRESENT128, Algorithm.FF1):
        cw = PayloadCodec(keys, alg).encode(USER, SESSION)
        acc = 0
        n = enc_trials if alg is Algorithm.PRESENT128 else enc_trials // 10
        for _ in range(n):
            acc += decode_or_none(PayloadCodec(with_keys(keys, enc_key=os.urandom(16)), alg), cw) is not None
        out.row(test=f"wrong enc_key only ({alg.name})", trials=n, identities_returned=acc,
                note="key commitment in the MAC: expected 0 (was ~1/256 without it)")
    msgs = set()
    for bits in (rng.integers(0, 2, 188, dtype=np.uint8), PayloadCodec(with_keys(keys, mac_key=os.urandom(16)),
                                                                         Algorithm.PRESENT128).encode(1, 1)):
        try:
            codec.decode(bits)
        except NoValidMark as e:
            msgs.add(str(e))
    out.row(test="distinct error messages", trials=2, identities_returned=0, note=" | ".join(sorted(msgs)))
    tag = os.urandom(4)
    first, last = bytes([tag[0] ^ 1]) + tag[1:], tag[:3] + bytes([tag[3] ^ 1])
    res = {}
    for name, other in (("differs in byte 0", first), ("differs in byte 3", last), ("equal", tag)):
        t = time.perf_counter_ns()
        for _ in range(300_000):
            hmac.compare_digest(tag, other)
        res[name] = (time.perf_counter_ns() - t) / 300_000
    out.row(test="compare_digest timing (ns/call)", trials=300_000, identities_returned=0,
            note="; ".join(f"{k}: {v:.1f}" for k, v in res.items()))
    out.close()


if __name__ == "__main__":
    bit_level()
    video_level()
