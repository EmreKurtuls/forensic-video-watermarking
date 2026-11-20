import subprocess, sys, os, numpy as np
W, H, F = 352, 288, 30
YS, CS = W * H, (W // 2) * (H // 2)
FS = YS + 2 * CS
src, work = sys.argv[1], sys.argv[2]
os.makedirs(work, exist_ok=True)
raw = os.path.join(work, "orig.yuv")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-frames:v", str(F), "-f", "rawvideo", "-pix_fmt", "yuv420p", raw], check=True)
buf = np.fromfile(raw, np.uint8).reshape(F, FS)
Y0 = buf[:, :YS].reshape(F, H, W).astype(np.float64)
rng = np.random.default_rng(1)

def roundtrip(Y, mode):
    out = buf.copy(); out[:, :YS] = np.clip(np.rint(Y), 0, 255).astype(np.uint8).reshape(F, YS)
    p_in, p_enc, p_out = [os.path.join(work, n) for n in ("in.yuv", "enc.mkv", "out.yuv")]
    out.tofile(p_in)
    fmt = ["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{W}x{H}"]
    if mode == "ffv1":
        codec = ["-c:v", "ffv1"]
    else:
        codec = ["-c:v", "libx264", "-crf", str(mode), "-preset", "medium"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", *fmt, "-i", p_in, *codec, p_enc], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", p_enc, *fmt[:4], p_out], check=True)
    sz = os.path.getsize(p_enc)
    return np.fromfile(p_out, np.uint8).reshape(F, FS)[:, :YS].reshape(F, H, W).astype(np.float64), sz

def psnr(a, b):
    return 10 * np.log10(255 ** 2 / np.mean((a - b) ** 2))

N = 8
C = np.array([[np.sqrt((1 if k == 0 else 2) / N) * np.cos(np.pi * (2 * n + 1) * k / (2 * N)) for n in range(N)] for k in range(N)])
def blocks(Y):
    return Y.reshape(F, H // 8, 8, W // 8, 8).transpose(0, 1, 3, 2, 4)
def unblocks(B):
    return B.transpose(0, 1, 3, 2, 4).reshape(F, H, W)
def dct(B): return C @ B @ C.T
def idct(D): return C.T @ D @ C

POS = (2, 3)
NB = (H // 8) * (W // 8)
ID_BITS = 96

def qim_embed(Y, bits, delta):
    D = dct(blocks(Y))
    c = D[..., POS[0], POS[1]].reshape(-1)
    d = bits * delta / 2
    c = delta * np.round((c - d) / delta) + d
    D[..., POS[0], POS[1]] = c.reshape(F, H // 8, W // 8)
    return unblocks(idct(D))

def qim_extract(Y, delta):
    c = dct(blocks(Y))[..., POS[0], POS[1]].reshape(-1)
    e0 = np.abs(c - delta * np.round(c / delta))
    e1 = np.abs(c - (delta * np.round((c - delta / 2) / delta) + delta / 2))
    return (e1 < e0).astype(int)

def lsb_embed(Y, bits):
    Yi = Y.astype(int).reshape(-1)
    need = (Yi & 1) != bits
    step = rng.choice([-1, 1], size=Yi.size)
    step[Yi == 0] = 1; step[Yi == 255] = -1
    Yi = np.where(need, Yi + step, Yi)
    return Yi.reshape(F, H, W).astype(np.float64)

def id_recovery(rx_bits, msg):
    n = len(rx_bits) // ID_BITS * ID_BITS
    votes = rx_bits[:n].reshape(-1, ID_BITS).mean(axis=0)
    return int(np.sum((votes > 0.5).astype(int) != msg))

attacks = ["ffv1", 18, 23, 28, 35]
msg = rng.integers(0, 2, ID_BITS)
_, base = roundtrip(Y0, 23)
print(f"clip: foreman CIF {W}x{H}, {F} frames (1 s). baseline x264 crf23 size {base/1024:.0f} KB")

print("\n[LSB matching, every Y pixel]  capacity/frame = %d bits = %.1f KB" % (YS, YS / 8 / 1024))
bits = np.tile(msg, YS * F // ID_BITS + 1)[:YS * F]
Ys = lsb_embed(Y0, bits.reshape(F, H, W).reshape(-1))
print(f"  PSNR(Y) = {psnr(Y0, Ys):.1f} dB")
for a in attacks:
    Yr, _ = roundtrip(Ys, a)
    rx = (Yr.astype(int) & 1).reshape(-1)
    print(f"  after {'FFV1 lossless' if a=='ffv1' else 'x264 crf'+str(a):16s} BER={np.mean(rx != bits):.4f}  96-bit ID wrong bits after voting={id_recovery(rx, msg)}")

for delta in (8, 16, 32):
    print(f"\n[DCT-QIM blind, 1 coef/8x8 block, delta={delta}]  capacity/frame = {NB} bits = {NB/8:.0f} bytes")
    bits = np.tile(msg, NB * F // ID_BITS + 1)[:NB * F]
    Ys = qim_embed(Y0, bits, delta)
    Ys = np.clip(np.rint(Ys), 0, 255)
    print(f"  PSNR(Y) = {psnr(Y0, Ys):.1f} dB")
    for a in attacks:
        Yr, _ = roundtrip(Ys, a)
        rx = qim_extract(Yr, delta)
        print(f"  after {'FFV1 lossless' if a=='ffv1' else 'x264 crf'+str(a):16s} BER={np.mean(rx != bits):.4f}  96-bit ID wrong bits after voting={id_recovery(rx, msg)}")
