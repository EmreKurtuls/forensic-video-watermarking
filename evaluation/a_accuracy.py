import os

import numpy as np

from common import (ALGS, CLIP_1080, CLIPS, CSV, FORMATS, SESSION, USER, WORK, Timer, available, clip_path,
                    embed_ss, extract_ss, film_setup)
from videowm import bulk
from videowm.embed import lsb
from videowm.video_io import probe


def small():
    out = CSV("a_small", ["clip", "format", "algorithm", "identified", "raw_bit_errors", "ber", "embed_s", "extract_s"])
    for clip in available(CLIPS + [CLIP_1080]):
        src = clip_path(clip)
        for alg_name, alg in ALGS.items():
            keys, codec = film_setup(clip, alg)
            cw = codec.encode(USER, SESSION)
            for ext in FORMATS:
                dst = WORK / f"a_{clip}_{alg_name}{ext}"
                with Timer() as te:
                    embed_ss(src, dst, cw, keys)
                with Timer() as tx:
                    ident, bits, _ = extract_ss(dst, src, codec)
                ok = ident is not None and (ident.user_id, ident.session) == (USER, SESSION)
                err = int(np.sum(bits != cw))
                out.row(clip=clip, format=ext[1:], algorithm=alg_name, identified=ok, raw_bit_errors=err,
                        ber=err / len(cw), embed_s=te.wall, extract_s=tx.wall)
                os.remove(dst)
    out.close()


def large():
    out = CSV("a_large", ["clip", "format", "algorithm", "payload_bytes", "capacity_bytes", "fits", "exact",
                          "encrypt_s", "embed_s", "extract_s", "decrypt_s"])
    for clip in available(["foreman_cif", "bbb_action_720p"]):
        src = clip_path(clip)
        info = probe(src, count_frames=True)
        cap = lsb.capacity_bits(info.width, info.height, info.nframes) // 8
        keys, _ = film_setup(clip)
        for size in (1024, 64 * 1024, 1024 * 1024):
            data = os.urandom(size)
            for alg_name, enc, dec in (("PRESENT-128 CTR", bulk.present_ctr_encrypt, bulk.present_ctr_decrypt),
                                       ("FF1 64-bit chunks", bulk.ff1_encrypt, bulk.ff1_decrypt)):
                with Timer() as tenc:
                    blob = enc(keys.enc_key, data)
                for ext in FORMATS:
                    dst = WORK / f"al_{clip}{ext}"
                    with Timer() as te:
                        lsb.embed_video(src, dst, blob, keys.stego_key)
                    with Timer() as tx:
                        back = lsb.extract_bytes(dst, len(blob))
                    with Timer() as tdec:
                        plain = dec(keys.enc_key, back)
                    out.row(clip=clip, format=ext[1:], algorithm=alg_name, payload_bytes=size, capacity_bytes=cap,
                            fits=len(blob) <= cap, exact=plain == data, encrypt_s=tenc.wall, embed_s=te.wall,
                            extract_s=tx.wall, decrypt_s=tdec.wall)
                    os.remove(dst)
    out.close()


if __name__ == "__main__":
    small()
    large()
