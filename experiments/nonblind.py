import sys, numpy as np
sys.argv = [sys.argv[0], "dl/foreman_part.y4m", "work"]
exec(open("stdm.py").read().split("msg = rng")[0])
_, c0 = coefs(Y0)
def ss_embed(Y, bits, alpha):
    D, c = coefs(Y); s = (2 * bits - 1)[:, None]
    c = c + alpha * s * U * np.sqrt(len(ZZ))
    D[..., idx[0], idx[1]] = c.reshape(F, H // 8, W // 8, len(ZZ)); return unblocks(idct(D))
def ss_extract(Y):
    _, c = coefs(Y); return (np.sum((c - c0) * U, axis=1) > 0).astype(int)
msg = rng.integers(0, 2, ID_BITS); attacks = ["ffv1", 23, 28, 35, 40]
for alpha in (1, 2, 4):
    bits = np.tile(msg, NB * F // ID_BITS + 1)[:NB * F]
    Ys = np.clip(np.rint(ss_embed(Y0, bits, alpha)), 0, 255)
    print(f"[Non-blind spread spectrum, alpha={alpha}] PSNR(Y)={psnr(Y0, Ys):.1f} dB")
    for a in attacks:
        Yr, sz = roundtrip(Ys, a); rx = ss_extract(Yr)
        print(f"   {'FFV1 lossless' if a=='ffv1' else 'x264 crf'+str(a):14s} ({sz/1024:5.0f} KB) BER={np.mean(rx != bits):.3f}  ID wrong bits={id_recovery(rx, msg)}")
