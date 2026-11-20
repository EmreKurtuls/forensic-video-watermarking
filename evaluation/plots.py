import csv
import os
from collections import defaultdict

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS, WORK

INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
SERIES = {"SS a=3.1": ("#2a78d6", "o", "-"), "QIM D=24": ("#eb6834", "s", "--"),
          "QIM D=64": ("#1baf7a", "^", "-."), "LSB": ("#eda100", "D", ":")}
ALG = {"PRESENT-128": "#2a78d6", "FF1": "#eb6834"}

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "lines.linewidth": 1.5, "lines.markersize": 5, "savefig.dpi": 300, "savefig.bbox": "tight"})


def load(name):
    p = RESULTS / f"{name}.csv"
    if not p.exists():
        return None
    with open(p) as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def save(fig, name):
    fig.savefig(RESULTS / f"{name}.png")
    plt.close(fig)
    print("wrote", RESULTS / f"{name}.png")


def fig_tradeoff():
    rows = load("e_quality")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for method, label in (("SS", "SS a=3.1"), ("QIM", "QIM D=24")):
        by = defaultdict(list)
        for r in rows:
            if r["method"] == method:
                by[float(r["strength"])].append((num(r["psnr_y"]), num(r["raw_errors_crf28"]) / 188))
        ks = sorted(by)
        x = [np.mean([a for a, _ in by[k]]) for k in ks]
        y = [np.mean([b for _, b in by[k]]) for k in ks]
        c, m, ls = SERIES[label]
        name = "Non-blind SS" if method == "SS" else "Blind QIM"
        ax.plot(x, y, color=c, marker=m, ls=ls, label=name)
        dy = 5 if method == "SS" else -10
        for k, xi, yi in zip(ks, x, y):
            ax.annotate(f"{'a' if method == 'SS' else 'D'}={k:g}", (xi, yi), textcoords="offset points",
                        xytext=(-6, dy), fontsize=5.5, color=c)
    ax.axvline(45, color=MUTED, lw=0.8, ls=":")
    ax.text(44.7, ax.get_ylim()[1] * 0.45, "threshold 45 dB", fontsize=6, color=MUTED, rotation=90, ha="right")
    ax.set_ylim(bottom=-0.04)
    ax.set_xlabel("PSNR(Y) of the marked copy [dB]")
    ax.set_ylabel("raw BER after H.264 CRF 28")
    ax.legend(loc="upper left")
    save(fig, "fig_e_tradeoff")


def fig_ber_vs_crf():
    rows = load("b_degradation")
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5), sharey=True)
    for ax, codec in zip(axes, ("H.264", "H.265")):
        for method, (c, m, ls) in SERIES.items():
            xs, ys = [], []
            for crf in (18, 23, 28, 35, 40):
                v = [num(r["raw_ber"]) for r in rows if r["method"] == method and r["attack"] == f"{codec} CRF {crf}"]
                v = [a for a in v if not np.isnan(a)]
                if v:
                    xs.append(crf)
                    ys.append(np.mean(v))
            ax.plot(xs, ys, color=c, marker=m, ls=ls, label=method)
        ax.set_title(codec, fontsize=8, color=INK)
        ax.set_xlabel("CRF of the pirate re-encode")
        ax.set_xticks((18, 23, 28, 35, 40))
    axes[0].set_ylabel("raw BER (mean over clips)")
    axes[1].legend(loc="lower right")
    save(fig, "fig_b_ber_vs_crf")


def fig_identified_matrix():
    rows = load("b_degradation")
    if not rows:
        return
    attacks = list(dict.fromkeys(r["attack"] for r in rows))
    methods = list(SERIES)
    M = np.full((len(attacks), len(methods)), np.nan)
    for i, a in enumerate(attacks):
        for j, m in enumerate(methods):
            v = [r["identified"] == "True" for r in rows if r["attack"] == a and r["method"] == m]
            if v:
                M[i, j] = np.mean(v)
    fig, ax = plt.subplots(figsize=(4.2, 0.22 * len(attacks) + 0.8))
    ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(attacks)):
        for j in range(len(methods)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=6,
                        color="white" if M[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(methods)), methods)
    ax.set_yticks(range(len(attacks)), attacks)
    ax.tick_params(length=0)
    ax.grid(False)
    ax.xaxis.tick_top()
    ax.set_title("share of clips where the user was identified", fontsize=8, color=INK, pad=18)
    save(fig, "fig_b_identified")


def fig_crypto():
    rows = load("d_crypto")
    if not rows:
        return
    ops = [o for o in dict.fromkeys(r["operation"] for r in rows) if not o.startswith("bulk")]
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    h = 0.38
    for k, (alg, c) in enumerate(ALG.items()):
        v = [num(next(r["us_per_op"] for r in rows if r["algorithm"] == alg and r["operation"] == o)) for o in ops]
        y = np.arange(len(ops)) + (k - 0.5) * h
        ax.barh(y, v, height=h * 0.9, color=c, label=alg, hatch=None if k == 0 else "////", edgecolor="white", lw=0.5)
        for yi, vi in zip(y, v):
            ax.text(vi * 1.1, yi, f"{vi:.0f}", va="center", fontsize=6, color=INK)
    ax.set_yticks(range(len(ops)), ops)
    ax.set_xscale("log")
    ax.set_xlim(5, 600)
    ax.set_xticks([10, 30, 100, 300], ["10", "30", "100", "300"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("time per operation [µs] (log scale)")
    ax.invert_yaxis()
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, "fig_d_crypto")


def fig_histograms(clip="bbb_action_720p"):
    p = WORK.parent / "g_histograms.npz"
    if not p.exists():
        return
    z = np.load(p)
    if f"{clip}_orig" not in z:
        clip = sorted(k[:-5] for k in z.files if k.endswith("_orig"))[0]
    o, m, r = z[f"{clip}_orig"], z[f"{clip}_marked"], z[f"{clip}_crf18"]
    fig, axes = plt.subplots(2, 3, figsize=(7, 3.4), sharex=True, layout="constrained")
    for c, ch in enumerate("RGB"):
        axes[0, c].plot(o[c] / o[c].sum(), color=MUTED, lw=1.2, label="original")
        axes[0, c].plot(m[c] / m[c].sum(), color="#2a78d6", lw=0.8, ls="--", label="marked (SS)")
        axes[0, c].set_title(ch, fontsize=8, color=INK)
        axes[1, c].plot((m[c] - o[c]) / o[c].sum(), color="#2a78d6", lw=0.8, label="marked - original")
        axes[1, c].plot((r[c] - o[c]) / o[c].sum(), color="#eb6834", lw=0.8, ls=":", label="CRF 18 - original")
        axes[1, c].set_xlabel("pixel value")
    axes[0, 0].set_ylabel("share of pixels")
    axes[1, 0].set_ylabel("difference")
    axes[0, 2].legend(loc="upper right", fontsize=6)
    axes[1, 2].legend(loc="upper right", fontsize=6)
    fig.suptitle(f"RGB histograms, {clip}", fontsize=8, color=INK)
    save(fig, "fig_g_histograms")


def fig_capacity():
    rows = load("f_capacity")
    if not rows:
        return
    methods = ["SS", "QIM", "LSB"]
    channels = ["lossless", "H.264 CRF 23", "H.264 CRF 28"]
    fig, ax = plt.subplots(figsize=(3.5, 2.3), layout="constrained")
    w = 0.26
    colors = ["#2a78d6", "#eb6834", "#eda100"]
    for k, (m, c) in enumerate(zip(methods, colors)):
        v = [np.mean([num(r["bsc_capacity_bits_per_s"]) for r in rows if r["method"] == m and r["channel"] == ch] or [np.nan])
             for ch in channels]
        raw = v
        v = [max(a, 0.5) for a in v]
        for xi, a in zip(np.arange(3) + (k - 1) * w, raw):
            if a == 0:
                ax.text(xi, 0.7, "0", ha="center", fontsize=6, color=INK)
        ax.bar(np.arange(3) + (k - 1) * w, v, width=w * 0.9, color=c, label=m, edgecolor="white", lw=0.5,
               hatch=[None, "////", "...."][k])
    ax.set_yscale("log")
    ax.set_xticks(range(3), channels)
    ax.set_ylabel("capacity [bit/s] (log)")
    ax.legend(loc="upper right")
    save(fig, "fig_f_capacity")


if __name__ == "__main__":
    for f in (fig_tradeoff, fig_ber_vs_crf, fig_identified_matrix, fig_crypto, fig_histograms, fig_capacity):
        f()
