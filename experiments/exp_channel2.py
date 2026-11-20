import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from wm import *
from skimage.metrics import structural_similarity as ssim
clips = {"foreman": "dl/foreman_cif.y4m", "akiyo": "dl/akiyo_cif.y4m"}
variants = [("G1", 1, False), ("G1-mask", 1, True), ("G8", 8, False), ("G30", 30, False), ("G30-mask", 30, True)]
attacks = [23, 28, 35, 40]
KEY = b"\x07" * 32; rng = np.random.default_rng(5)
for name, path in clips.items():
    w, h = probe(path); frames = list(reader(path, w, h)); F = len(frames); nb = (h//8)*(w//8)
    for vname, G, mask in variants:
        ng = F // G
        signs = rng.choice([-1.0, 1.0], size=(ng, nb)); Us = [pattern(KEY, g, nb) for g in range(ng)]
        marked = []
        for i, fr in enumerate(frames[: ng*G]):
            g = i // G; Y = y_of(fr, w, h)
            a = texture_alpha(Y, 2.0) if mask else 2.0
            f2 = fr.copy(); f2[: w*h] = embed_frame(Y, signs[g], Us[g], a).reshape(-1); marked.append(f2)
        ps = np.mean([psnr(y_of(a, w, h), y_of(b, w, h)) for a, b in zip(frames, marked)])
        ss = np.mean([ssim(y_of(a, w, h), y_of(b, w, h), data_range=255) for a, b in zip(frames[::10], marked[::10])])
        line = f"{name:8s} {vname:9s} PSNR={ps:4.1f} SSIM={ss:.4f} |"
        for crf in attacks:
            out = f"work/{name}_{vname}_{crf}.mp4"
            p = writer(out, w, h, ["-c:v", "libx264", "-crf", str(crf), "-preset", "medium"], fps=30)
            for f2 in marked: p.stdin.write(f2.tobytes())
            p.stdin.close(); p.wait()
            z = np.zeros((ng, nb))
            for i, (r, o) in enumerate(zip(reader(out, w, h), frames)):
                g = i // G; z[g] += corr_frame(y_of(r, w, h), y_of(o, w, h), Us[g]) * signs[g]
            np.save(f"res/z2_{name}_{vname}_{crf}.npy", z)
            line += f" crf{crf}: BER={np.mean(z<0):.3f}"
        print(line, flush=True)
