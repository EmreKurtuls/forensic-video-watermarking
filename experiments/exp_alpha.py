import itertools, os, subprocess, sys
import numpy as np
from skimage.metrics import structural_similarity as ssim
from videowm.embed.ss_nonblind import embed_video
from videowm.extract import identify
from videowm.keys import derive_film_keys, generate_master_key
from videowm.payload import Algorithm, PayloadCodec
from videowm.video_io import read_frames

ALPHAS = (1.0, 1.5, 2.0, 3.1, 4.5, 6.0, 9.0)
out = "data/out/exp_alpha"; os.makedirs(out, exist_ok=True)
for clip in sys.argv[1:]:
    src = f"data/clips/{clip}.mkv"
    keys = derive_film_keys(generate_master_key(), clip.encode()); codec = PayloadCodec(keys, Algorithm.PRESENT128)
    cw = codec.encode(1001, 7)
    for a in ALPHAS:
        m = f"{out}/{clip}.mkv"; st = embed_video(src, m, cw, keys.stego_key, alpha=a)
        pairs = list(itertools.islice(zip(read_frames(src), read_frames(m)), 0, None, 12))
        ss = np.mean([ssim(x.y, y.y, data_range=255) for x, y in pairs])
        ssmin = np.min([ssim(x.y, y.y, data_range=255) for x, y in pairs])
        row = []
        for crf in (28, 35, 40):
            p = f"{out}/{clip}.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", m, "-c:v", "libx264", "-crf", str(crf), "-preset", "fast", p], check=True)
            ident, ex = identify(p, src, codec)
            row.append(f"{int(np.sum(ex.bits != cw)):3d}{'✅' if ident and ident.user_id == 1001 else '❌'}")
        print(f"{clip}\t{a}\t{st.psnr_y:.1f}\t{ss:.4f}\t{ssmin:.4f}\t" + "\t".join(row), flush=True)
