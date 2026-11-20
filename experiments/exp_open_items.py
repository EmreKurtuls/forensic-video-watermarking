import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "experiments"))
from common import CLIPS, SESSION, available, clip_path, embed_ss, film_setup
from videowm import attacks
from videowm.embed.ss_nonblind import group_length
from videowm.extract import identify
from videowm.video_io import Frame, VideoWriter, probe, read_frames

WORK = ROOT / "data" / "out" / "exp_open"
STEG = ROOT / "data" / "out" / "exp_steg"
WORK.mkdir(parents=True, exist_ok=True)


def vmaf(dist, ref):
    out = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(dist), "-i", str(ref), "-lavfi",
                          "[0:v]settb=1/1000,setpts=N*40[d];[1:v]settb=1/1000,setpts=N*40[r];[d][r]libvmaf=n_threads=4",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    return float(re.findall(r"VMAF score: ([0-9.]+)", out)[-1])


def part_vmaf():
    print(f"{'clip':18s} {'marked':>8s} {'CRF18':>8s} {'CRF23':>8s} {'marked+CRF23':>13s}")
    for clip in available(CLIPS):
        src = clip_path(clip)
        marked = STEG / f"{clip}_marked.mkv"
        if not marked.exists():
            keys, codec = film_setup(clip)
            embed_ss(src, marked, codec.encode(1001, SESSION), keys)
        row = [vmaf(marked, src), vmaf(STEG / f"{clip}_orig_crf18.mp4", src), vmaf(STEG / f"{clip}_orig_crf23.mp4", src),
               vmaf(STEG / f"{clip}_marked_crf23.mp4", src)]
        print(f"{clip:18s} " + " ".join(f"{v:8.2f}" for v in row[:3]) + f" {row[3]:13.2f}", flush=True)


def combine(copies, dst, mode, seg):
    info = probe(copies[0])
    readers = [read_frames(c) for c in copies]
    with VideoWriter(dst, info.width, info.height, info.fps, color=info.color) as wr:
        for i, frs in enumerate(zip(*readers)):
            if mode == "average":
                y = np.rint(np.mean([f.y.astype(np.float64) for f in frs], axis=0)).astype(np.uint8)
                wr.write(Frame(y, frs[0].u, frs[0].v))
            else:
                wr.write(frs[(i // seg) % len(frs)])


def part_collusion():
    users = [1001, 1002, 1003, 1004]
    for clip in ("bbb_action_720p", "tos_static_720p"):
        src = clip_path(clip)
        keys, codec = film_setup(clip)
        cws = {u: codec.encode(u, SESSION) for u in users}
        copies = {}
        for u in users[:3]:
            copies[u] = WORK / f"{clip}_{u}.mkv"
            if not copies[u].exists():
                embed_ss(src, copies[u], cws[u], keys)
        seg = group_length(probe(src).fps)
        for k in (2, 3):
            for mode in ("average", "segments"):
                col = WORK / f"{clip}_{mode}{k}.mkv"
                combine([copies[u] for u in users[:k]], col, mode, seg)
                pirate = attacks.reencode(col, WORK / f"{clip}_{mode}{k}_crf23.mp4", 23)
                ident, ex = identify(pirate, src, codec)
                sc = {u: float(np.sum(ex.soft * (cws[u] * 2.0 - 1.0)) / np.sum(np.abs(ex.soft))) for u in users}
                errs = {u: int(np.sum(ex.bits != cws[u])) for u in users[:k]}
                named = "nobody (NoValidMark)" if ident is None else f"user {ident.user_id}"
                print(f"{clip:16s} {k} colluders {mode:8s}: names {named:22s} bit errors vs each {errs}  "
                      f"correlation {', '.join(f'{u}: {v:+.2f}' for u, v in sc.items())} (1004 innocent)", flush=True)
                col.unlink(); Path(pirate).unlink()


def norm_r(d):
    r = np.full(len(d), np.nan)
    r[2:-1] = d[2:-1] / np.maximum((d[1:-2] + d[3:]) / 2, 1e-9)
    return r


def gap_stat(d, G):
    r = norm_r(d)
    ok = np.flatnonzero(~np.isnan(r))
    k = len(ok) // G
    top = np.sort(ok[np.argsort(r[ok])[-k:]])
    gaps = np.diff(top)
    J = round(G / 4)
    return float(np.mean((gaps >= G - J) & (gaps <= G + J))), float(np.mean(gaps == G))


def part_gaps():
    from exp_steganalysis import features
    src = ROOT / "data/raw/BigBuckBunny_320x180.mp4"
    G = group_length(probe(src).fps)
    keys, codec = film_setup("bbb_full")
    marked = WORK / "bbb_full_keyed.mkv"
    embed_ss(src, marked, codec.encode(1001, SESSION), keys)
    keyed = attacks.reencode(marked, WORK / "bbb_full_keyed_crf23.mp4", 23)
    marked.unlink()
    rows = {"unmarked CRF 23": np.load(STEG / "bbb_full_orig_crf23_feat.npz")["d"],
            "fixed groups CRF 23": np.load(STEG / "bbb_full_marked_crf23_feat.npz")["d"],
            "keyed groups CRF 23": features(keyed)["d"]}
    print(f"top 1/{G} frames by normalised temporal energy; share of gaps in [{G - round(G / 4)}, {G + round(G / 4)}] and exactly {G}")
    for name, d in rows.items():
        a, b = gap_stat(d, G)
        print(f"  {name:22s} in-range {a:.3f}   exactly G {b:.3f}", flush=True)
    rng = np.random.default_rng(0)
    sims = []
    for _ in range(200):
        n = len(rows["unmarked CRF 23"])
        top = np.sort(rng.choice(n, n // G, replace=False))
        g = np.diff(top)
        sims.append(np.mean((g >= G - round(G / 4)) & (g <= G + round(G / 4))))
    print(f"  random positions        in-range {np.mean(sims):.3f} +- {np.std(sims):.3f}")


T = 3


def _spam_dir(x, axis_slices):
    d0, d1, d2 = (np.clip(s, -T, T) + T for s in axis_slices(x))
    idx = (d0 * 49 + d1 * 7 + d2).ravel()
    c3 = np.bincount(idx, minlength=343).reshape(7, 7, 7).astype(np.float64)
    c2 = c3.sum(axis=2, keepdims=True)
    return (c3 / np.maximum(c2, 1)).ravel()


def spam(y):
    x = y.astype(np.int32)
    h = x[:, :-1] - x[:, 1:]
    v = x[:-1, :] - x[1:, :]
    dg = x[:-1, :-1] - x[1:, 1:]
    ad = x[:-1, 1:] - x[1:, :-1]
    S = lambda dd, a: {
        "r": lambda _: (dd[:, :-2], dd[:, 1:-1], dd[:, 2:]),
        "l": lambda _: (-dd[:, 2:], -dd[:, 1:-1], -dd[:, :-2]),
        "d": lambda _: (dd[:-2, :], dd[1:-1, :], dd[2:, :]),
        "u": lambda _: (-dd[2:, :], -dd[1:-1, :], -dd[:-2, :]),
        "dr": lambda _: (dd[:-2, :-2], dd[1:-1, 1:-1], dd[2:, 2:]),
        "ul": lambda _: (-dd[2:, 2:], -dd[1:-1, 1:-1], -dd[:-2, :-2]),
        "dl": lambda _: (dd[:-2, 2:], dd[1:-1, 1:-1], dd[2:, :-2]),
        "ur": lambda _: (-dd[2:, :-2], -dd[1:-1, 1:-1], -dd[:-2, 2:]),
    }[a]
    straight = np.mean([_spam_dir(None, S(h, "r")), _spam_dir(None, S(h, "l")), _spam_dir(None, S(v, "d")), _spam_dir(None, S(v, "u"))], axis=0)
    diag = np.mean([_spam_dir(None, S(dg, "dr")), _spam_dir(None, S(dg, "ul")), _spam_dir(None, S(ad, "dl")), _spam_dir(None, S(ad, "ur"))], axis=0)
    return np.concatenate([straight, diag])


def frame_feats(path, step=12):
    return np.array([spam(f.y) for i, f in enumerate(read_frames(path)) if i % step == 0])


def fld_train(X0, X1, lam=1e-3):
    mu = X0.mean(0), X1.mean(0)
    Sw = np.cov(X0, rowvar=False) + np.cov(X1, rowvar=False)
    Sw += lam * np.trace(Sw) / len(Sw) * np.eye(len(Sw))
    w = np.linalg.solve(Sw, mu[1] - mu[0])
    return w, float(w @ (mu[0] + mu[1]) / 2)


def auc(neg, pos):
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def part_spam():
    clips = available(CLIPS)
    for cond, orig_of, mark_of in (("lossless", lambda c: clip_path(c), lambda c: STEG / f"{c}_marked.mkv"),
                                   ("CRF 23", lambda c: STEG / f"{c}_orig_crf23.mp4", lambda c: STEG / f"{c}_marked_crf23.mp4")):
        F = {c: (frame_feats(orig_of(c)), frame_feats(mark_of(c))) for c in clips}
        accs, aucs = [], []
        for test in clips:
            X0 = np.vstack([F[c][0] for c in clips if c != test]); X1 = np.vstack([F[c][1] for c in clips if c != test])
            w, b = fld_train(X0, X1)
            s0, s1 = F[test][0] @ w, F[test][1] @ w
            accs.append(np.mean(np.concatenate([s0 < b, s1 >= b])))
            aucs.append(auc(s0, s1))
            print(f"  {cond:8s} test {test:18s} accuracy {accs[-1]:.2f}  AUC {aucs[-1]:.2f}", flush=True)
        print(f"== {cond}: mean accuracy {np.mean(accs):.2f}, mean AUC {np.mean(aucs):.2f} (0.5 = chance)", flush=True)


if __name__ == "__main__":
    {"vmaf": part_vmaf, "collusion": part_collusion, "gaps": part_gaps, "spam": part_spam}[sys.argv[1]]()
