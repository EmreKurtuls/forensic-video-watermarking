import numpy as np

N = 8
_k = np.arange(N)
C = np.sqrt(2.0 / N) * np.cos(np.pi * (2 * _k[None, :] + 1) * _k[:, None] / (2 * N))
C[0] /= np.sqrt(2.0)


def _zigzag(n: int = N) -> list[tuple[int, int]]:
    order = []
    for s in range(2 * n - 1):
        diag = [(i, s - i) for i in range(n) if 0 <= s - i < n]
        order += diag if s % 2 else diag[::-1]
    return order


ZIGZAG = _zigzag()
LOW_BAND = tuple(range(1, 6))


def band_positions(band=LOW_BAND):
    rows = np.array([ZIGZAG[i][0] for i in band])
    cols = np.array([ZIGZAG[i][1] for i in band])
    return rows, cols


def block_grid(height: int, width: int) -> tuple[int, int]:
    return height // N, width // N


def block_dct(y: np.ndarray) -> np.ndarray:
    bh, bw = block_grid(*y.shape)
    b = y[:bh * N, :bw * N].astype(np.float64).reshape(bh, N, bw, N).transpose(0, 2, 1, 3)
    return C @ b @ C.T


def block_idct(d: np.ndarray, like: np.ndarray) -> np.ndarray:
    bh, bw = d.shape[:2]
    pix = (C.T @ d @ C).transpose(0, 2, 1, 3).reshape(bh * N, bw * N)
    out = like.copy()
    out[:bh * N, :bw * N] = np.clip(np.rint(pix), 0, 255).astype(np.uint8)
    return out


def get_band(d: np.ndarray, band=LOW_BAND) -> np.ndarray:
    r, c = band_positions(band)
    return d[:, :, r, c].reshape(-1, len(band))


def set_band(d: np.ndarray, values: np.ndarray, band=LOW_BAND) -> np.ndarray:
    r, c = band_positions(band)
    out = d.copy()
    out[:, :, r, c] = values.reshape(d.shape[0], d.shape[1], len(band))
    return out
