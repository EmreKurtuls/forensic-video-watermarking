import itertools
import json
import subprocess
from pathlib import Path

import numpy as np

from math import ceil

from videowm.ciphers import ff1 as F1
from videowm.ciphers import present as P
from videowm.ciphers import siphash24
from videowm.dct import LOW_BAND, ZIGZAG, block_dct, get_band
from videowm.embed.ss_nonblind import DEFAULT_ALPHA, GroupMap, bit_index, embed_luma, masked_alpha
from videowm.keys import derive_film_keys, generate_master_key, pattern
from videowm.metrics import psnr
from videowm.payload import Algorithm, PayloadCodec, pack_block
from videowm.video_io import read_frames

OUT = Path("presentation/watermark_animation")
CLIP, FRAME, FILM, UID, SESSION = "data/clips/tos_static_720p.mkv", 10, b"film-0239", 42, 0

master = generate_master_key()
keys = derive_film_keys(master, FILM)
block = pack_block(UID, SESSION)
bits = lambda v, n: format(v, f"0{n}b")
hx = lambda bts: bts.hex()


def present_states():
    cipher = P.PRESENT128(keys.enc_key)
    rks, states = cipher._rk, [block]
    for rk in rks[:31]:
        s_in, s = states[-1] ^ rk, 0
        for j in range(8):
            s |= P._SP[j][(s_in >> (8 * j)) & 255]
        states.append(s)
    states.append(states[-1] ^ rks[31])
    assert states[-1] == cipher.encrypt_int(block)
    return states


def ff1_states():
    f = F1.FF1(keys.enc_key, radix=2)
    r, n, tweak = 2, 64, keys.film_id
    t, u, v = len(tweak), 32, 32
    a, b = block >> 32, block & 0xFFFFFFFF
    bl = ceil((r ** v - 1).bit_length() / 8)
    d = 4 * ceil(bl / 4) + 4
    Pb = bytes([1, 2, 1]) + r.to_bytes(3, "big") + bytes([10, u % 256]) + n.to_bytes(4, "big") + t.to_bytes(4, "big")
    head = Pb + tweak + b"\x00" * ((-t - bl - 1) % 16)
    states = [block]
    for i in range(10):
        R = f._prf(head + bytes([i]) + b.to_bytes(bl, "big"))
        S = R.to_bytes(16, "big")
        for j in range(1, ceil(d / 16)):
            S += f._aes.encrypt_int(R ^ j).to_bytes(16, "big")
        y = int.from_bytes(S[:d], "big")
        a, b = b, (a + y) % (r ** u if i % 2 == 0 else r ** v)
        states.append((a << 32) | b)
    assert states[-1] == f.encrypt_int(block, 64, tweak)
    return states


fr = next(itertools.islice(read_frames(CLIP), FRAME, None))
H, W = fr.y.shape
nb = (H // 8) * (W // 8)
bw = W // 8
g = GroupMap(keys.stego_key, 24).group(FRAME)
u = pattern(keys.stego_key, g, nb, len(LOW_BAND))
d = block_dct(fr.y)
alpha = masked_alpha(d, DEFAULT_ALPHA)
cand = [(r, c) for r in range(21, 31) for c in range(39, 50)]
r, c = max(cand, key=lambda rc: alpha[rc[0] * bw + rc[1]])
b = r * bw + c
bidx = bit_index(g, nb, 188)


def to_png(y, name):
    raw = y.tobytes() + fr.u.tobytes() + fr.v.tobytes()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{W}x{H}", "-i", "-",
                    "-frames:v", "1", str(OUT / name)], input=raw, check=True)


from skimage.metrics import structural_similarity

to_png(fr.y, "original.png")
algs = {}
for label, alg, slug, states in (("PRESENT-128", Algorithm.PRESENT128, "present", present_states()),
                                 ("FF1", Algorithm.FF1, "ff1", ff1_states())):
    codec = PayloadCodec(keys, alg)
    cw_bytes = codec.encode_bytes(UID, SESSION)
    cw = np.unpackbits(np.frombuffer(cw_bytes, np.uint8))[:188]
    assert cw_bytes[:8] == states[-1].to_bytes(8, "big")
    ym = embed_luma(fr.y, cw[bidx] * 2.0 - 1.0, u, DEFAULT_ALPHA)
    to_png(ym, f"marked_{slug}.png")
    diff = np.clip(128 + 20 * (ym.astype(int) - fr.y.astype(int)), 0, 255).astype(np.uint8)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-i", "-",
                    str(OUT / f"diff20_{slug}.png")], input=diff.tobytes(), check=True)
    algs[label] = {
        "slug": slug, "rounds": len(states) - 2 if alg is Algorithm.PRESENT128 else len(states) - 1,
        "states": [bits(x, 64) for x in states],
        "ciphertext_hex": hx(cw_bytes[:8]), "tag_hex": hx(cw_bytes[8:12]), "kc": hx(codec._kc),
        "codeword": "".join(map(str, cw)),
        "close_bit": int(cw[bidx[b]]),
        "pix_m": ym[r * 8:r * 8 + 8, c * 8:c * 8 + 8].astype(int).tolist(),
        "coef_m": np.round(block_dct(ym)[r, c], 1).tolist(),
        "psnr_y": round(psnr(fr.y, ym), 2),
        "ssim_y": round(float(structural_similarity(fr.y, ym, data_range=255)), 4),
        "max_abs_change": int(np.max(np.abs(ym.astype(int) - fr.y.astype(int)))),
        "share_changed": round(float(np.mean(ym != fr.y)), 3),
    }
    print(label, {k: algs[label][k] for k in ("ciphertext_hex", "tag_hex", "psnr_y", "ssim_y", "max_abs_change", "close_bit")})

data = {
    "film_id": FILM.decode(), "user_id": UID, "session": SESSION, "version": 1,
    "master_key": hx(master), "enc_key": hx(keys.enc_key), "mac_key": hx(keys.mac_key), "stego_key": hx(keys.stego_key),
    "block_bits": bits(block, 64),
    "width": W, "height": H, "blocks_x": bw, "blocks_y": H // 8, "nblocks": nb,
    "alpha": round(float(DEFAULT_ALPHA), 3),
    "close": {
        "row": r, "col": c, "block": b, "bit_index": int(bidx[b]),
        "alpha_b": round(float(alpha[b]), 3), "u": [int(x) for x in u[b]],
        "band": [list(ZIGZAG[i]) for i in LOW_BAND],
        "pix_o": fr.y[r * 8:r * 8 + 8, c * 8:c * 8 + 8].astype(int).tolist(),
        "coef_o": np.round(d[r, c], 1).tolist(),
    },
    "algs": algs,
}
(OUT / "data.json").write_text(json.dumps(data))
