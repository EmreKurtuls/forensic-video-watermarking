import json
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest


def line(name, k, n):
    t = binomtest(k, n, 0.5, alternative="greater")
    lo, hi = binomtest(k, n, 0.5).proportion_ci(0.95, method="exact")
    return f"{name:20s} {k:3d}/{n:<3d} = {k / n:5.1%}   95 % CI [{lo:.0%}, {hi:.0%}]   p(better than chance) = {t.pvalue:.3f}"


def main(folder):
    files = sorted(Path(folder).glob("2afc_*.json"))
    if not files:
        sys.exit(f"no 2afc_*.json files in {folder}")
    per_clip, per_person, all_ = defaultdict(lambda: [0, 0]), {}, [0, 0]
    for f in files:
        d = json.loads(f.read_text())
        ok = [t["correct"] for t in d["trials"]]
        per_person[f"{d['participant']} ({f.name})"] = (sum(ok), len(ok))
        for t in d["trials"]:
            per_clip[t["clip"]][0] += t["correct"]
            per_clip[t["clip"]][1] += 1
        all_[0] += sum(ok); all_[1] += len(ok)
    print(f"{len(files)} participants\n")
    print(line("all answers", *all_))
    print("\nper clip")
    for c, (k, n) in sorted(per_clip.items()):
        print(line(c, k, n))
    print("\nper participant")
    for p, (k, n) in per_person.items():
        print(line(p[:20], k, n))
    lo = binomtest(*all_, 0.5).proportion_ci(0.95, method="exact").high
    print(f"\nInvisible at the 2AFC threshold if the upper 95 % bound ({lo:.0%}) stays below 75 %.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "subjective_test/results")
