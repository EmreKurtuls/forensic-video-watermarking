import itertools, os, subprocess, sys, time
import numpy as np
from skimage.metrics import structural_similarity as ssim
from videowm.embed.ss_nonblind import embed_video
from videowm.extract import identify
from videowm.keys import derive_film_keys, generate_master_key
from videowm.payload import Algorithm, PayloadCodec
from videowm.video_io import probe, read_frames

src = sys.argv[1]; out = "data/out/exp_resolution"; os.makedirs(out, exist_ok=True)
info = probe(src, count_frames=True); secs = info.nframes / float(info.fps)
def ff(*a): subprocess.run(["ffmpeg", "-v", "error", "-y", *a], check=True)
masters = {"1080p": f"{out}/m1080.mkv", "720p": f"{out}/m720.mkv"}
ff("-i", src, "-c:v", "ffv1", masters["1080p"])
ff("-i", src, "-vf", "scale=1280:720:flags=bicubic", "-c:v", "ffv1", masters["720p"])
print(f"source {info.width}x{info.height} {float(info.fps):g} fps, {info.nframes} frames ({secs:.1f} s)")
keys = derive_film_keys(generate_master_key(), b"CROWD"); codec = PayloadCodec(keys, Algorithm.PRESENT128); cw = codec.encode(1001, 7)
for res, m in masters.items():
    i2 = probe(m); W, H = i2.width, i2.height
    t = time.perf_counter(); st = embed_video(m, f"{out}/w{res}.mkv", cw, keys.stego_key); te = time.perf_counter() - t
    ss = np.mean([ssim(a.y, b.y, data_range=255) for a, b in itertools.islice(zip(read_frames(m), read_frames(f"{out}/w{res}.mkv")), 0, None, 10)])
    t = time.perf_counter(); ident, ex = identify(f"{out}/w{res}.mkv", m, codec); tx = time.perf_counter() - t
    sz = os.path.getsize(f"{out}/w{res}.mkv") / 2**20
    print(f"[{res}] {W}x{H}, {st.nblocks} blocks/frame, PSNR {st.psnr_y:.1f} dB, SSIM {ss:.4f}, "
          f"embed {te:.1f} s ({te/secs:.1f} s per video-second), extract {tx:.1f} s ({tx/secs:.1f} s/s), lossless file {sz:.0f} MB")
    for name, crf, vf in (("CRF23", 23, None), ("CRF28", 28, None), ("CRF35", 35, None), ("CRF40", 40, None),
                          ("50%+CRF28", 28, f"scale={W//2}:{H//2}")):
        p = f"{out}/p{res}.mp4"
        ff("-i", f"{out}/w{res}.mkv", *(["-vf", vf] if vf else []), "-c:v", "libx264", "-crf", str(crf), "-preset", "fast", p)
        ident, ex = identify(p, m, codec)
        print(f"     {name:10s} raw errors {int(np.sum(ex.bits != cw)):3d}/188  {'ID ✅' if ident and ident.user_id == 1001 else 'ID ❌'}", flush=True)
