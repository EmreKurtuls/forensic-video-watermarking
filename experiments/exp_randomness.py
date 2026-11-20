import random
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from videowm import keys

STS = ROOT / "data/raw/sts_dl/sts-2.1.2/sts-2.1.2"
WORK = ROOT / "data/out/sts"
N_BITS, N_STREAMS = 1_000_000, 100
N_BYTES = N_BITS * N_STREAMS // 8


def gen_secrets():
    return b"".join(keys.generate_master_key() for _ in range(N_BYTES // keys.MASTER_KEY_LEN + 1))


def gen_hkdf():
    m, out, i = keys.generate_master_key(), [], 0
    while sum(map(len, out)) < N_BYTES:
        k = keys.derive_film_keys(m, f"film-{i}".encode())
        out.append(k.enc_key + k.mac_key + k.stego_key)
        i += 1
    return b"".join(out)


def gen_shake():
    sk = keys.derive_film_keys(keys.generate_master_key(), b"film-0").stego_key
    nb, K = (720 // 8) * (1280 // 8), 5
    out, g = [], 0
    while sum(map(len, out)) < N_BYTES:
        out.append(np.packbits(keys.pattern(sk, g, nb, K).ravel() > 0).tobytes())
        g += 1
    return b"".join(out)


def gen_mt19937():
    return random.Random(2026).randbytes(N_BYTES)


def gen_lcg():
    x, out = 1, bytearray(N_BYTES)
    for i in range(N_BYTES):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        out[i] = x & 0xFF
    return bytes(out)


SOURCES = {"secrets": gen_secrets, "hkdf": gen_hkdf, "shake": gen_shake, "mt19937": gen_mt19937, "lcg": gen_lcg}


def run(name):
    d = WORK / name
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(STS, d, ignore=shutil.ignore_patterns("obj", "src", "*.o"))
    (d / "input.bin").write_bytes(SOURCES[name]()[:N_BYTES])
    stdin = f"0\ninput.bin\n1\n0\n{N_STREAMS}\n1\n"
    subprocess.run(["./assess", str(N_BITS)], cwd=d, input=stdin, text=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rep = (d / "experiments/AlgorithmTesting/finalAnalysisReport.txt").read_text()
    shutil.copy(d / "experiments/AlgorithmTesting/finalAnalysisReport.txt", WORK / f"{name}_report.txt")
    (d / "input.bin").unlink()
    return name, rep


def summarise(rep):
    res = {}
    for line in rep.splitlines():
        m = re.match(r"\s*(\d+\s+){10}(\S+)\s+\*?\s*(\d+)/(\d+)\s*\*?\s+(\w+)", line)
        if m:
            t = m.group(5)
            n, f = res.get(t, (0, 0))
            res[t] = (n + 1, f + ("*" in line))
    return res


def main():
    reuse = "--reuse" in sys.argv
    names = [a for a in sys.argv[1:] if a != "--reuse"] or list(SOURCES)
    WORK.mkdir(parents=True, exist_ok=True)
    if reuse:
        reports = {n: (WORK / f"{n}_report.txt").read_text() for n in names}
    else:
        with ThreadPoolExecutor(len(names)) as ex:
            reports = dict(ex.map(run, names))
    tests = sorted({t for r in reports.values() for t in summarise(r)})
    with open(WORK / "summary.csv", "w") as f:
        f.write("test," + ",".join(names) + "\n")
        print(f"{'test':26s}" + "".join(f"{n:>12s}" for n in names))
        for t in tests:
            cells = []
            for n in names:
                k, fail = summarise(reports[n]).get(t, (0, 0))
                cells.append(f"{k - fail}/{k}")
            print(f"{t:26s}" + "".join(f"{c:>12s}" for c in cells))
            f.write(t + "," + ",".join(cells) + "\n")


if __name__ == "__main__":
    main()
