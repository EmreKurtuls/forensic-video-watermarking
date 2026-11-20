import subprocess
import sys
from pathlib import Path

from videowm.embed.ss_nonblind import DEFAULT_ALPHA, embed_video
from videowm.keys import derive_film_keys, generate_master_key
from videowm.payload import Algorithm, PayloadCodec

src = Path(sys.argv[1])
uid = int(sys.argv[2]) if len(sys.argv) > 2 else 1001
alpha = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_ALPHA
out = Path("data/out/compare")
out.mkdir(parents=True, exist_ok=True)
keys = derive_film_keys(generate_master_key(), src.stem.encode())
marked = out / f"{src.stem}_marked.mkv"
st = embed_video(src, marked, PayloadCodec(keys, Algorithm.PRESENT128).encode(uid, 0), keys.stego_key, alpha)
label = lambda t: f"drawtext=text='{t}':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=6"
fc = (f"[0:v]scale=640:-2,{label('ORIGINAL')}[o];"
      f"[1:v]scale=640:-2,{label(f'MARKED  user {uid}  PSNR {st.psnr_y:.1f} dB')}[m];"
      f"[1:v][0:v]blend=all_expr='clip(128+20*(A-B),0,255)',scale=640:-2,{label('DIFFERENCE x20')}[d];"
      f"[o][m][d]hstack=inputs=3[v]")
for name, extra in (("compare.mp4", ["-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p"]),
                    ("still.png", ["-vf", "select=eq(n\\,60)", "-frames:v", "1"])):
    args = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-i", str(marked), "-filter_complex", fc, "-map", "[v]"]
    if name.endswith("png"):
        args = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-i", str(marked), "-filter_complex",
                fc.replace("[v]", "[s];[s]select=eq(n\\,60)[v]"), "-map", "[v]", "-frames:v", "1"]
    else:
        args += extra
    subprocess.run(args + [str(out / f"{src.stem}_{name}")], check=True)
print(f"PSNR(Y) {st.psnr_y:.1f} dB -> {out / (src.stem + '_compare.mp4')}")
