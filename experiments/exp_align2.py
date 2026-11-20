import sys, os, subprocess, numpy as np, bchlib
sys.path.insert(0, os.path.dirname(__file__))
import wm
from wm import probe, reader, y_of, pattern, embed_frame, corr_frame, texture_alpha, psnr
G, KEY = 30, b"\x09" * 32
bch = bchlib.BCH(12, m=8); rng = np.random.default_rng(3)
BANDS = {"mid (zigzag 1-12)": [(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),(2,1),(3,0),(4,0),(3,1),(2,2)],
         "low (zigzag 1-5)":  [(0,1),(1,0),(2,0),(1,1),(0,2)]}
def thumb(Y):
    h, w = Y.shape; return Y[: h//9*9, : w//11*11].reshape(h//9, 9, w//11, 11).mean((1, 3))
for clip in ("foreman", "akiyo"):
    path = f"dl/{clip}_cif.y4m"; w, h = probe(path); frames = list(reader(path, w, h)); F = len(frames); nb = (h//8)*(w//8)
    O = [y_of(f, w, h) for f in frames]; To = np.stack([thumb(y.astype(float)) for y in O])
    data = rng.bytes(12); cw = np.unpackbits(np.frombuffer(data + bch.encode(data), np.uint8)).astype(int)
    bitidx = np.arange((F // G) * nb).reshape(F // G, nb) % 192
    keep = [i for i in range(20, F) if i % 10 != 0]; sel = "+".join(f"eq(n\\,{i})" for i in keep)
    for bname, zz in BANDS.items():
        wm.ZZ = zz; wm.K = len(zz); wm.ZI = (np.array([p[0] for p in zz]), np.array([p[1] for p in zz]))
        alpha = 2.0 * np.sqrt(12 / len(zz))
        marked = []
        for i, fr in enumerate(frames):
            g = i // G; Y = O[i]
            f2 = fr.copy(); f2[: w*h] = embed_frame(Y, cw[bitidx[g]] * 2.0 - 1, pattern(KEY, g, nb), texture_alpha(Y, alpha)).reshape(-1); marked.append(f2)
        ps = np.mean([psnr(O[i], y_of(m, w, h)) for i, m in enumerate(marked)])
        raw = f"work/al2_{clip}.yuv"
        with open(raw, "wb") as f:
            for f2 in marked: f.write(f2.tobytes())
        for sname, scale in (("trim+drop", None), ("trim+drop+scale0.75", "264:216"), ("trim+drop+scale0.5", "176:144")):
            pir = f"work/al2_{clip}_p.mp4"
            vf = f"select='{sel}',setpts=N/30/TB" + (f",scale={scale}" if scale else "")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}", "-r", "30", "-i", raw,
                            "-vf", vf, "-fps_mode", "passthrough", "-c:v", "libx264", "-crf", "23", pir], check=True)
            P = [y_of(f, w, h) for f in reader(pir, w, h, vf=f"scale={w}:{h}")]
            match, prev = [], -1
            for t in (thumb(y.astype(float)) for y in P):
                cand = list(range(F)) if prev < 0 else list(range(min(prev + 1, F - 1), min(F, prev + 6)))
                cost = [np.mean((To[j] - t) ** 2) + (0 if prev < 0 else 2.0 * (j - prev - 1)) for j in cand]
                prev = cand[int(np.argmin(cost))]; match.append(prev)
            out = []
            for mode, mp in (("truth", keep), ("aligned", match)):
                soft = np.zeros(192)
                for k, j in enumerate(mp):
                    g = j // G; np.add.at(soft, bitidx[g], corr_frame(P[k], O[j], pattern(KEY, g, nb)))
                bits = (soft > 0).astype(np.uint8); by = np.packbits(bits).tobytes()
                d, e = bytearray(by[:12]), bytearray(by[12:]); n = bch.decode(bytes(d), bytes(e))
                if n >= 0: bch.correct(d, e)
                out.append(f"{mode}: {int(np.sum(bits != cw)):3d} err {'OK ' if n >= 0 and bytes(d) == data else 'FAIL'}")
            grp_acc = np.mean(np.array(match) // G == np.array(keep) // G)
            print(f"{clip:8s} {bname:18s} PSNR={ps:4.1f} {sname:20s} group-acc={grp_acc:.3f} | " + " | ".join(out), flush=True)
