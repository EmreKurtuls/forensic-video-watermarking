import numpy as np, itertools
G = {"G1": 1, "G1-mask": 1, "G8": 8, "G30": 30, "G30-mask": 30}
schemes = {
    "REP-96 (ECC yok)": (96, "bit", 0),
    "BCH(192,96) t=12": (192, "bit", 12),
    "BCH(285,96) t=21": (285, "bit", 21),
    "RS(24,12)": (192, "byte", 6),
    "RS(36,12)": (288, "byte", 12),
}
Ts = [1, 2, 3, 4, 5, 6, 8, 10]
rng = np.random.default_rng(9); TR = 300
def ok_rate(units, n, kind, cap, soft):
    succ = 0
    m = len(units) // n * n
    for _ in range(TR):
        u = rng.permutation(units)[:m].reshape(-1, n)
        comb = u.sum(0) if soft else np.sign(u).sum(0)
        err = comb <= 0
        e = err.sum() if kind == "bit" else err.reshape(-1, 8).any(1).sum()
        succ += e <= cap
    return succ / TR
for clip in ("foreman", "akiyo"):
    for crf in (23, 28, 35):
        print(f"\n== {clip}, x264 CRF {crf}: ID'yi %99 okumak icin gereken min sure (s), soft combining [hard] ==")
        print(f"{'scheme':20s}" + "".join(f"{v:>14s}" for v in G))
        for sname, (n, kind, cap) in schemes.items():
            row = f"{sname:20s}"
            for v, g in G.items():
                z = np.load(f"res/z2_{clip}_{v}_{crf}.npy")
                res = []
                for soft in (True, False):
                    best = ">10"
                    for T in Ts:
                        units = z[: T * 30 // g].reshape(-1)
                        if len(units) >= n and ok_rate(units, n, kind, cap, soft) >= 0.99:
                            best = str(T); break
                    res.append(best)
                row += f"{res[0]+' ['+res[1]+']':>14s}"
            print(row, flush=True)
