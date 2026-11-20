import sys, numpy as np
sys.argv = [sys.argv[0], "dl/foreman_part.y4m", "work"]
exec(open("hiding.py").read().split("attacks = [")[0])
ZZ = [(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),(2,1),(3,0),(4,0),(3,1),(2,2)]
key = np.random.default_rng(2026)
U = key.choice([-1.0, 1.0], size=(NB * F, len(ZZ))); U /= np.linalg.norm(U, axis=1, keepdims=True)
idx = (tuple(p[0] for p in ZZ), tuple(p[1] for p in ZZ))
def coefs(Y):
    D = dct(blocks(Y)); return D, D[..., idx[0], idx[1]].reshape(NB * F, len(ZZ))
def stdm_embed(Y, bits, delta):
    D, c = coefs(Y); p = np.sum(c * U, axis=1); d = bits * delta / 2
    q = delta * np.round((p - d) / delta) + d
    c = c + (q - p)[:, None] * U
    D[..., idx[0], idx[1]] = c.reshape(F, H // 8, W // 8, len(ZZ)); return unblocks(idct(D))
def stdm_extract(Y, delta):
    _, c = coefs(Y); p = np.sum(c * U, axis=1)
    e0 = np.abs(p - delta * np.round(p / delta)); e1 = np.abs(p - (delta * np.round((p - delta / 2) / delta) + delta / 2))
    return (e1 < e0).astype(int)
msg = rng.integers(0, 2, ID_BITS); attacks = ["ffv1", 18, 23, 28, 35]
for delta in (16, 32, 48, 64):
    bits = np.tile(msg, NB * F // ID_BITS + 1)[:NB * F]
    Ys = np.clip(np.rint(stdm_embed(Y0, bits, delta)), 0, 255)
    print(f"[STDM blind, 12 coefs/block, delta={delta}] PSNR(Y)={psnr(Y0, Ys):.1f} dB")
    for a in attacks:
        Yr, _ = roundtrip(Ys, a); rx = stdm_extract(Yr, delta)
        print(f"   {'FFV1 lossless' if a=='ffv1' else 'x264 crf'+str(a):14s} BER={np.mean(rx != bits):.3f}  ID wrong bits={id_recovery(rx, msg)}")
