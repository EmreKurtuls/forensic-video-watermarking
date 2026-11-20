import sys, os, numpy as np, subprocess
sys.path.insert(0, os.path.dirname(__file__))
from wm import *

clips = {"foreman": "dl/foreman_cif.y4m", "akiyo": "dl/akiyo_cif.y4m"}
attacks = [23, 28, 35]
KEY = b"\x07" * 32
os.makedirs("work", exist_ok=True); os.makedirs("res", exist_ok=True)
rng = np.random.default_rng(5)
for name, path in clips.items():
    w, h = probe(path)
    frames = list(reader(path, w, h))
    F = len(frames); nb = (h // 8) * (w // 8)
    signs = rng.choice([-1.0, 1.0], size=(F, nb))
    Us = [pattern(KEY, i, nb) for i in range(F)]
    for variant in ("uniform", "masked"):
        marked = []
        for i, fr in enumerate(frames):
            Y = y_of(fr, w, h)
            a = 2.0 if variant == "uniform" else texture_alpha(Y, 2.0)
            Ym = embed_frame(Y, signs[i], Us[i], a)
            f2 = fr.copy(); f2[: w*h] = Ym.reshape(-1); marked.append(f2)
        ps = np.mean([psnr(y_of(a, w, h), y_of(b, w, h)) for a, b in zip(frames, marked)])
        for crf in attacks:
            out = f"work/{name}_{variant}_{crf}.mp4"
            p = writer(out, w, h, ["-c:v", "libx264", "-crf", str(crf), "-preset", "medium"], fps=30)
            for f2 in marked: p.stdin.write(f2.tobytes())
            p.stdin.close(); p.wait()
            z = np.stack([corr_frame(y_of(r, w, h), y_of(o, w, h), Us[i]) * signs[i]
                          for i, (r, o) in enumerate(zip(reader(out, w, h), frames))])
            kb = os.path.getsize(out) / 1024
            np.save(f"res/z_{name}_{variant}_{crf}.npy", z)
            print(f"{name:8s} {variant:8s} PSNR={ps:5.1f} dB  crf{crf} ({kb:5.0f} KB/10s)  raw block BER={np.mean(z<0):.3f}")
