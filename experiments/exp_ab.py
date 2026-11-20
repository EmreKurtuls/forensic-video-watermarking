import sys, os, time, glob, subprocess, hashlib, numpy as np, bchlib
sys.path.insert(0, os.path.dirname(__file__))
from wm import *
from ciphers import present128_round_keys, present_encrypt_block, siphash24
from skimage.metrics import structural_similarity as ssim

SEG, MAXF, ALPHA = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
SRC = "dl/bbb/BigBuckBunny_320x180.mp4"; W, H = 320, 180
tag = f"ab_seg{SEG}_a{ALPHA:g}"; wd = f"work/{tag}"; os.makedirs(wd, exist_ok=True)
master = hashlib.sha256(b"demo master key (experiment only)").digest()
kd = lambda info: hashlib.sha256(master + info).digest()
enc_key, mac_key, stego_key = kd(b"enc")[:16], kd(b"mac")[:16], kd(b"stego")
rks = present128_round_keys(int.from_bytes(enc_key, "big"))
bch = bchlib.BCH(12, m=8)
FILM_ID, ALG = b"BBB-0001", b"\x01"

def make_codeword(user_id, session):
    block = (1 << 60) | (user_id << 28) | (session << 4)
    c = present_encrypt_block(block, rks).to_bytes(8, "big")
    t = (siphash24(mac_key, ALG + FILM_ID + c) & 0xFFFFFFFF).to_bytes(4, "big")
    data = c + t
    cw = data + bch.encode(data)
    return np.unpackbits(np.frombuffer(cw, np.uint8)).astype(int)

def read_codeword(bits):
    by = np.packbits(bits.astype(np.uint8)).tobytes()
    data, ecc = bytearray(by[:12]), bytearray(by[12:24])
    nerr = bch.decode(bytes(data), bytes(ecc))
    if nerr < 0: return None, "BCH decode failed"
    bch.correct(data, ecc)
    c, t = bytes(data[:8]), bytes(data[8:12])
    exp = (siphash24(mac_key, ALG + FILM_ID + c) & 0xFFFFFFFF).to_bytes(4, "big")
    if not __import__("hmac").compare_digest(t, exp): return None, "MAC invalid -> no valid mark"
    return c, f"MAC ok ({nerr} bit corrected)"

t0 = time.time()
nb = (H // 8) * (W // 8)
nseg = MAXF // SEG
codec = ["-c:v", "libx264", "-crf", "20", "-preset", "medium", "-x264-params", f"keyint={SEG}:min-keyint={SEG}:scenecut=0"]
pa, pb = writer(f"{wd}/A.mp4", W, H, codec), writer(f"{wd}/B.mp4", W, H, codec)
ps, ss = [], []
for i, fr in enumerate(reader(SRC, W, H)):
    if i >= nseg * SEG: break
    s = i // SEG; Y = y_of(fr, W, H); U = pattern(stego_key, s, nb); a = texture_alpha(Y, ALPHA)
    for p, sign in ((pa, -1.0), (pb, 1.0)):
        Ym = embed_frame(Y, np.full(nb, sign), U, a)
        f2 = fr.copy(); f2[: W*H] = Ym.reshape(-1); p.stdin.write(f2.tobytes())
    if i % 97 == 0: ps.append(psnr(Y, Ym)); ss.append(ssim(Y, Ym, data_range=255))
for p in (pa, pb): p.stdin.close(); p.wait()
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", SRC, "-frames:v", str(nseg*SEG), "-an", "-c:v", "libx264", "-crf", "20", "-preset", "medium", f"{wd}/ref_default_gop.mp4"], check=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", SRC, "-frames:v", str(nseg*SEG), "-an", *codec, f"{wd}/ref_seg_gop.mp4"], check=True)
sz = {k: os.path.getsize(f"{wd}/{k}.mp4") for k in ("A", "B", "ref_default_gop", "ref_seg_gop")}
print(f"[{tag}] segments={nseg} ({SEG/24:.2f} s each), video={nseg*SEG/24:.0f} s, PSNR={np.mean(ps):.1f} dB SSIM={np.mean(ss):.4f}, embed+encode {time.time()-t0:.0f}s")
print(f"   bitrate: default GOP {sz['ref_default_gop']/1024:.0f} KB, segment-aligned GOP {sz['ref_seg_gop']/1024:.0f} KB "
      f"(+{100*(sz['ref_seg_gop']/sz['ref_default_gop']-1):.1f}%), marked A {sz['A']/1024:.0f} KB")
seglist = ",".join(str(k*SEG) for k in range(1, nseg))
for v in ("A", "B"):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{wd}/{v}.mp4", "-c", "copy", "-f", "segment", "-segment_format", "mpegts",
                    "-segment_frames", seglist, f"{wd}/{v}_%04d.ts"], check=True)
print(f"   TS segments: A={len(glob.glob(wd+'/A_*.ts'))} B={len(glob.glob(wd+'/B_*.ts'))}")

users = {"user 1001 / session 7": (1001, 7), "user 2002 / session 3": (2002, 3)}
db = {}
for uname, (uid, sess) in users.items():
    cw = make_codeword(uid, sess)
    db[np.packbits(cw[:64].astype(np.uint8)).tobytes()] = uname
    with open(f"{wd}/user_{uid}.ts", "wb") as out:
        for s in range(nseg):
            with open(f"{wd}/{'B' if cw[s % 192] else 'A'}_{s:04d}.ts", "rb") as f: out.write(f.read())
    for crf in (28, 35, 40):
        pir = f"{wd}/pirate_{uid}_crf{crf}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{wd}/user_{uid}.ts", "-an", "-c:v", "libx264", "-crf", str(crf), pir], check=True)
        segcorr = np.zeros(nseg); nfr = 0
        for i, (r, o) in enumerate(zip(reader(pir, W, H), reader(SRC, W, H))):
            if i >= nseg * SEG: break
            s = i // SEG; segcorr[s] += np.sum(corr_frame(y_of(r, W, H), y_of(o, W, H), pattern(stego_key, s, nb))); nfr += 1
        soft = np.zeros(192)
        for s in range(nseg): soft[s % 192] += segcorr[s]
        bits = (soft > 0).astype(int)
        raw_err = int(np.sum((segcorr > 0).astype(int) != cw[np.arange(nseg) % 192]))
        c, msg = read_codeword(bits)
        who = db.get(c, "not in DB") if c else "-"
        print(f"   {uname}: pirate CRF {crf} ({os.path.getsize(pir)/1024:.0f} KB, {nfr} frames) "
              f"segment bit errors={raw_err}/{nseg}, {msg}, identified: {who}", flush=True)
print(f"   total {time.time()-t0:.0f}s")
