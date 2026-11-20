import hashlib, os, subprocess, numpy as np

ZZ = [(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),(2,1),(3,0),(4,0),(3,1),(2,2)]
ZI = (np.array([p[0] for p in ZZ]), np.array([p[1] for p in ZZ]))
K = len(ZZ)
N8 = 8
C = np.array([[np.sqrt((1 if k == 0 else 2) / N8) * np.cos(np.pi * (2*n+1) * k / (2*N8)) for n in range(N8)] for k in range(N8)])

def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout.strip()
    w, h = map(int, out.split(",")[:2]); return w, h

def reader(path, w, h, vf=None, start_frame=0):
    cmd = ["ffmpeg", "-v", "error", "-i", path]
    if vf: cmd += ["-vf", vf]
    cmd += ["-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    fs = w * h * 3 // 2
    i = 0
    while True:
        b = p.stdout.read(fs)
        if len(b) < fs: break
        if i >= start_frame: yield np.frombuffer(b, np.uint8)
        i += 1
    p.wait()

def writer(path, w, h, codec_args, fps=24):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", *codec_args, path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)

def y_of(frame, w, h):
    return frame[: w*h].reshape(h, w)

def pattern(stego_key: bytes, frame_idx: int, nblocks: int) -> np.ndarray:
    nbytes = (nblocks * K + 7) // 8
    raw = hashlib.shake_256(stego_key + frame_idx.to_bytes(4, "big")).digest(nbytes)
    bits = np.unpackbits(np.frombuffer(raw, np.uint8))[: nblocks * K]
    return (bits.astype(np.float64) * 2 - 1).reshape(nblocks, K)

def blocks_dct(Y):
    h8, w8 = Y.shape[0] // 8 * 8, Y.shape[1] // 8 * 8
    B = Y[:h8, :w8].astype(np.float64).reshape(h8//8, 8, w8//8, 8).transpose(0, 2, 1, 3)
    return C @ B @ C.T

def coefs(Y):
    D = blocks_dct(Y)
    return D[..., ZI[0], ZI[1]].reshape(-1, K)

def embed_frame(Y, signs, U, alpha):
    h8, w8 = Y.shape[0] // 8 * 8, Y.shape[1] // 8 * 8
    D = blocks_dct(Y)
    a = np.broadcast_to(np.asarray(alpha, float), signs.shape)
    delta = (a * signs)[:, None] * U
    D[..., ZI[0], ZI[1]] += delta.reshape(D.shape[0], D.shape[1], K)
    B = (C.T @ D @ C).transpose(0, 2, 1, 3).reshape(h8, w8)
    out = Y.copy()
    out[:h8, :w8] = np.clip(np.rint(B), 0, 255).astype(np.uint8)
    return out

def corr_frame(Yr, Yo, U):
    return np.sum((coefs(Yr) - coefs(Yo)) * U, axis=1)

def texture_alpha(Y, alpha, lo=0.5, hi=2.0):
    D = blocks_dct(Y)
    ac = np.sqrt(np.sum(D**2, axis=(2, 3)) - D[..., 0, 0]**2).reshape(-1) + 1e-6
    s = np.clip(ac / np.median(ac), lo, hi)
    s /= np.sqrt(np.mean(s**2))
    return alpha * s

def psnr(a, b):
    m = np.mean((a.astype(float) - b.astype(float))**2)
    return 99.0 if m == 0 else 10 * np.log10(255**2 / m)
