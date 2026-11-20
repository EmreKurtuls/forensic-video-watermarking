import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from common import CLIPS, SESSION, USER, available, clip_path, embed_ss, film_setup
from videowm import attacks
from videowm.dct import block_dct, get_band
from videowm.embed.ss_nonblind import group_length
from videowm.video_io import probe, read_frames

WORK = ROOT / "data" / "out" / "exp_steg"
WORK.mkdir(parents=True, exist_ok=True)


def features(path):
    d, ea, es, za, zs = [], [], [], [], []
    prev = None
    for f in read_frames(path):
        y = f.y.astype(np.float64)
        a = get_band(block_dct(y))
        s = get_band(block_dct(y[4:, 4:]))
        d.append(np.nan if prev is None else float(np.mean((a - prev) ** 2)))
        prev = a
        ea.append(float(np.mean(a ** 2)))
        es.append(float(np.mean(s ** 2)))
        za.append(float(np.mean(np.abs(a) < 0.5)))
        zs.append(float(np.mean(np.abs(s) < 0.5)))
    return {k: np.array(v) for k, v in dict(d=d, ea=ea, es=es, za=za, zs=zs).items()}


def temporal_stat(d, G):
    r = np.full(len(d), np.nan)
    r[2:-1] = d[2:-1] / np.maximum((d[1:-2] + d[3:]) / 2, 1e-9)
    t = np.arange(len(d))
    ok = ~np.isnan(r)
    m = np.array([np.nanmedian(r[(t % G == p) & ok]) for p in range(G)])
    p0 = mannwhitneyu(r[(t % G == 0) & ok], r[(t % G != 0) & ok], alternative="greater").pvalue
    return float(m.max() / np.median(m)), int(m.argmax()), float(p0)


def windows(F, G):
    n = len(F["ea"]) // G
    w = lambda x: x[:n * G].reshape(n, G).mean(axis=1)
    R = w(F["ea"]) / w(F["es"])
    Z = w(F["zs"]) / np.maximum(w(F["za"]), 1e-9)
    return R, Z


def auc(neg, pos):
    neg, pos = np.asarray(neg), np.asarray(pos)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def variants_for(tag, src, marked):
    v = {"orig": src, "marked": marked}
    for name, p in list(v.items()):
        for crf in (18, 23):
            q = WORK / f"{tag}_{name}_crf{crf}.mp4"
            if not q.exists():
                attacks.reencode(p, q, crf)
            v[f"{name}_crf{crf}"] = q
    return v


def analyse(tag, src, marked, G, rows, pooled):
    for name, p in variants_for(tag, src, marked).items():
        f = WORK / f"{tag}_{name}_feat.npz"
        F = dict(np.load(f)) if f.exists() else features(p)
        np.savez(f, **F)
        S, ph, p0 = temporal_stat(F["d"], G)
        R, Z = windows(F, G)
        rows.append((tag, name, S, ph, p0, float(R.mean()), float(Z.mean())))
        pooled.setdefault(name, []).append((R, Z))
        print(f"{tag:18s} {name:13s} S={S:6.3f} peak={ph:2d} p0={p0:.2g}  R={R.mean():.4f}  Z={Z.mean():.4f}", flush=True)


CONDS = (("", "lossless"), ("_crf18", "CRF 18"), ("_crf23", "CRF 23"))


def main():
    film = "--film" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--film"]
    rows, pooled = [], {}
    if film:
        src = ROOT / (args[0] if args else "data/raw/BigBuckBunny_320x180.mp4")
        tag = "bbb_full"
        keys, codec = film_setup(tag)
        marked = WORK / f"{tag}_marked.mkv"
        if not marked.exists():
            embed_ss(src, marked, codec.encode(USER, SESSION), keys)
        analyse(tag, src, marked, group_length(probe(src).fps), rows, pooled)
    else:
        for clip in available(args or CLIPS):
            src = clip_path(clip)
            keys, codec = film_setup(clip)
            marked = WORK / f"{clip}_marked.mkv"
            if not marked.exists():
                embed_ss(src, marked, codec.encode(USER, SESSION), keys)
            analyse(clip, src, marked, group_length(probe(src).fps), rows, pooled)
        print("\n== pooled AUC over 1 s windows, all clips, one threshold ==")
        for cond, label in CONDS:
            neg, pos = pooled[f"orig{cond}"], pooled[f"marked{cond}"]
            for i, feat in enumerate(("R", "Z")):
                a = auc(np.concatenate([x[i] for x in neg]), np.concatenate([x[i] for x in pos]))
                print(f"  {label:8s} {feat}: AUC = {a:.3f}")
    print("\n== temporal detector: known phase p < 0.01 (unmarked = false alarm, marked = detection) ==")
    for cond, label in CONDS:
        neg = [r for r in rows if r[1] == f"orig{cond}"]
        pos = [r for r in rows if r[1] == f"marked{cond}"]
        print(f"  {label:8s} unmarked {sum(r[4] < 0.01 for r in neg)}/{len(neg)}   marked {sum(r[4] < 0.01 for r in pos)}/{len(pos)}"
              f"   peak on boundary (unknown phase) {sum(r[3] == 0 for r in pos)}/{len(pos)}")
    with open(WORK / ("summary_film.csv" if film else "summary.csv"), "w") as f:
        f.write("clip,variant,S_temporal,peak_phase,p_known_phase,R_calib,Z_calib\n")
        for r in rows:
            f.write(",".join(f"{x:.4g}" if isinstance(x, float) else str(x) for x in r) + "\n")


if __name__ == "__main__":
    main()
