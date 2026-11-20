import sys, os, subprocess, numpy as np, bchlib
sys.path.insert(0, os.path.dirname(__file__))
from wm import *
G, ALPHA, KEY = 30, 2.0, b"\x09" * 32
bch = bchlib.BCH(12, m=8); rng = np.random.default_rng(3)
def thumb(Y):
    h, w = Y.shape; return Y[: h//9*9, : w//11*11].reshape(h//9, 9, w//11, 11).mean((1, 3))
for clip in ("foreman", "akiyo"):
    path = f"dl/{clip}_cif.y4m"; w, h = probe(path); frames = list(reader(path, w, h)); F = len(frames); nb = (h//8)*(w//8)
    data = rng.bytes(12); cw = np.unpackbits(np.frombuffer(data + bch.encode(data), np.uint8)).astype(int)
    bitidx = np.arange((F // G) * nb).reshape(F // G, nb) % 192
    marked = []
    for i, fr in enumerate(frames):
        g = i // G; Y = y_of(fr, w, h)
        f2 = fr.copy(); f2[: w*h] = embed_frame(Y, cw[bitidx[g]] * 2.0 - 1, pattern(KEY, g, nb), texture_alpha(Y, ALPHA)).reshape(-1); marked.append(f2)
    raw = f"work/al_{clip}_marked.yuv"
    with open(raw, "wb") as f:
        for f2 in marked: f.write(f2.tobytes())
    keep = [i for i in range(20, F) if i % 10 != 0]
    sel = "+".join(f"eq(n\\,{i})" for i in keep)
    for crf in (23, 28):
        pir = f"work/al_{clip}_pirate_{crf}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}", "-r", "30", "-i", raw,
                        "-vf", f"select='{sel}',setpts=N/30/TB,scale=176:144", "-fps_mode", "passthrough", "-c:v", "libx264", "-crf", str(crf), pir], check=True)
        P = [y_of(f, w, h) for f in reader(pir, w, h, vf=f"scale={w}:{h}")]
        O = [y_of(f, w, h) for f in frames]
        To = np.stack([thumb(y.astype(float)) for y in O]); Tp = [thumb(y.astype(float)) for y in P]
        match, prev = [], -1
        for k, t in enumerate(Tp):
            cand = range(F) if prev < 0 else range(min(prev + 1, F - 1), min(F, prev + 6))
            cost = [np.mean((To[j] - t) ** 2) + (0 if prev < 0 else 2.0 * (j - prev - 1)) for j in cand]
            prev = list(cand)[int(np.argmin(cost))]; match.append(prev)
        print(f"  pirate frames={len(P)} expected={len(keep)}"); acc = np.mean(np.array(match[:len(keep)]) == np.array(keep[: len(match)]))
        for mode, mp in (("naive (frame k <-> k)", list(range(len(P)))), ("aligned", match)):
            soft = np.zeros(192)
            for k, j in enumerate(mp):
                g = j // G; c = corr_frame(P[k], O[j], pattern(KEY, g, nb))
                np.add.at(soft, bitidx[g], c)
            bits = (soft > 0).astype(np.uint8); by = np.packbits(bits).tobytes()
            d, e = bytearray(by[:12]), bytearray(by[12:]); n = bch.decode(bytes(d), bytes(e))
            if n >= 0: bch.correct(d, e)
            ok = n >= 0 and bytes(d) == data
            print(f"{clip:8s} CRF{crf} {mode:22s} align-acc={'-' if mode.startswith('naive') else f'{acc:.3f}'} "
                  f"bit errors before ECC={int(np.sum(bits != cw))}/192  ID {'RECOVERED' if ok else 'FAILED'}")
