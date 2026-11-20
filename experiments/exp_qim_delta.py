import os, sys, subprocess, itertools, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from skimage.metrics import structural_similarity as ssim
from videowm.keys import generate_master_key, derive_film_keys
from videowm.payload import PayloadCodec, Algorithm, NoValidMark
from videowm.embed import qim_blind, ss_nonblind
from videowm.extract import identify
from videowm.video_io import read_frames
sp = sys.argv[1] if len(sys.argv) > 1 else "."
def enc(src, dst, crf, vf=None):
    subprocess.run(["ffmpeg","-v","error","-y","-i",src,*(["-vf",vf] if vf else []),"-c:v","libx264","-crf",str(crf),dst],check=True); return dst
def mssim(a, b):
    return np.mean([ssim(x.y, y.y, data_range=255) for x, y in itertools.islice(zip(read_frames(a), read_frames(b)), 0, 300, 15)])
CASES = [("CRF18",18,None),("CRF23",23,None),("CRF28",28,None),("CRF35",35,None),("75%+CRF23",23,"scale=264:216")]
for clip in ("foreman", "akiyo"):
    src = f"{sp}/dl/{clip}_cif.y4m"; keys = derive_film_keys(generate_master_key(), clip.encode())
    codec = PayloadCodec(keys, Algorithm.PRESENT128); cw = codec.encode(1001, 7)
    print(f"== {clip}  (cells: raw bit errors/188, ✅ = ID recovered)")
    print(f"{'method':18s} {'PSNR':>5s} {'SSIM':>6s} | " + " | ".join(f"{c[0]:>9s}" for c in CASES), flush=True)
    for label, delta in [("SS a=3.10", None)] + [(f"QIM D={d}", d) for d in (16, 24, 32, 48, 64, 96)]:
        m = f"{sp}/work/q_{clip}.mkv"
        st = ss_nonblind.embed_video(src, m, cw, keys.stego_key) if delta is None else qim_blind.embed_video(src, m, cw, keys.stego_key, delta)
        cells = []
        for name, crf, vf in CASES:
            p = enc(m, f"{sp}/work/q_{clip}_p.mp4", crf, vf)
            if delta is None:
                ident, ex = identify(p, src, codec); bits = ex.bits
            else:
                bits = (qim_blind.extract_soft(p, keys.stego_key, delta, 188, width=352, height=288) > 0).astype(np.uint8)
                try: ident = codec.decode(bits)
                except NoValidMark: ident = None
            ok = ident is not None and ident.user_id == 1001
            cells.append(f"{int(np.sum(bits != cw)):3d} {'✅' if ok else '❌'}")
        print(f"{label:18s} {st.psnr_y:5.1f} {mssim(src, m):6.4f} | " + " | ".join(f"{c:>9s}" for c in cells), flush=True)
