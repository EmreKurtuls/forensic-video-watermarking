import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import wm
from wm import probe, reader, y_of, pattern, embed_frame, texture_alpha, psnr
from skimage.metrics import structural_similarity as ssim
BANDS = {"mid (zigzag 1-12)": [(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),(2,1),(3,0),(4,0),(3,1),(2,2)],
         "low (zigzag 1-5)":  [(0,1),(1,0),(2,0),(1,1),(0,2)],
         "low-mid (zigzag 1-9)": [(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),(2,1),(3,0)]}
rng = np.random.default_rng(1)
for clip in ("foreman", "akiyo"):
    path = f"dl/{clip}_cif.y4m"; w, h = probe(path); O = [y_of(f, w, h) for f in list(reader(path, w, h))[::15]]; nb = (h//8)*(w//8)
    for bname, zz in BANDS.items():
        wm.ZZ = zz; wm.K = len(zz); wm.ZI = (np.array([p[0] for p in zz]), np.array([p[1] for p in zz]))
        a = 2.0 * np.sqrt(12 / len(zz)); P, S = [], []
        for i, Y in enumerate(O):
            Ym = embed_frame(Y, rng.choice([-1.0, 1.0], nb), pattern(b"k", i, nb), texture_alpha(Y, a))
            P.append(psnr(Y, Ym)); S.append(ssim(Y, Ym, data_range=255))
        print(f"{clip:8s} {bname:22s} PSNR={np.mean(P):.1f} SSIM={np.mean(S):.4f}")
