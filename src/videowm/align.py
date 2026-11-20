import numpy as np

THUMB = 32
SEARCH_AHEAD = 5
GAP_PENALTY = 2.0


def thumbnail(y: np.ndarray, size: int = THUMB) -> np.ndarray:
    h, w = y.shape
    ch, cw = h // size, w // size
    return y[:ch * size, :cw * size].astype(np.float64).reshape(size, ch, size, cw).mean(axis=(1, 3))


def align(pirate_thumbs, original_thumbs, ahead: int = SEARCH_AHEAD, gap_penalty: float = GAP_PENALTY) -> list[int]:
    O = np.asarray(original_thumbs, dtype=np.float32).reshape(len(original_thumbs), -1)
    P = np.asarray(pirate_thumbs, dtype=np.float32).reshape(len(pirate_thumbs), -1)
    F = len(O)
    n = min(len(P), F)
    if n == 0:
        return []
    P = P[:n]
    D = F - n
    cost = np.empty((n, D + 1), np.float32)
    for d in range(D + 1):
        cost[:, d] = np.mean((P - O[d:d + n]) ** 2, axis=1)
    best = cost[0].copy()
    back = np.zeros((n, D + 1), np.int32)
    steps = np.arange(min(ahead, D + 1))
    for k in range(1, n):
        cand = np.full((len(steps), D + 1), np.inf, np.float32)
        for s in steps:
            cand[s, s:] = best[:D + 1 - s] + gap_penalty * s
        arg = np.argmin(cand, axis=0)
        best = cand[arg, np.arange(D + 1)] + cost[k]
        back[k] = arg
    d = int(np.argmin(best))
    path = np.empty(n, np.int64)
    for k in range(n - 1, -1, -1):
        path[k] = k + d
        d -= int(back[k, d])
    return path.tolist()
