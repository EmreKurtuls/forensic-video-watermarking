import os
import resource
import timeit
import tracemalloc

from common import ALGS, CLIP_1080, CLIPS, CSV, SESSION, USER, WORK, Timer, available, clip_path, embed_ss, extract_ss, film_setup
from videowm import bulk
from videowm.ciphers import FF1, PRESENT128
from videowm.video_io import probe


def crypto(repeats=5):
    out = CSV("d_crypto", ["algorithm", "operation", "us_per_op", "mb_per_s", "cpu_s", "peak_mem_kb"])
    key = os.urandom(16)
    x = 0x1000003E90000070
    p, f = PRESENT128(key), FF1(key, 2)
    ops = {
        "PRESENT-128": {"encrypt block": lambda: p.encrypt_int(x), "decrypt block": lambda: p.decrypt_int(x)},
        "FF1": {"encrypt block": lambda: f.encrypt_int(x, 64, b"FILM"), "decrypt block": lambda: f.decrypt_int(x, 64, b"FILM")},
    }
    for name, alg in ALGS.items():
        keys, codec = film_setup("PERF", alg)
        bits = codec.encode(USER, SESSION)
        ops[name]["payload encode (enc+MAC+BCH)"] = lambda c=codec: c.encode(USER, SESSION)
        ops[name]["payload decode (BCH+MAC+dec)"] = lambda c=codec, b=bits: c.decode(b)
        for op, fn in ops[name].items():
            n = 2000
            best = min(timeit.repeat(fn, number=n, repeat=repeats)) / n
            out.row(algorithm=name, operation=op, us_per_op=best * 1e6, mb_per_s="", cpu_s="", peak_mem_kb="")
    data, small = os.urandom(1024 * 1024), os.urandom(64 * 1024)
    for name, enc, dec in (("PRESENT-128", bulk.present_ctr_encrypt, bulk.present_ctr_decrypt),
                           ("FF1", bulk.ff1_encrypt, bulk.ff1_decrypt)):
        for op, d in (("bulk encrypt 1 MB", data), ("bulk decrypt 1 MB", data)):
            blob = enc(key, d) if "decrypt" in op else None
            fn = (lambda: dec(key, blob)) if blob is not None else (lambda: enc(key, d))
            with Timer() as t:
                fn()
            sb = enc(key, small) if blob is not None else None
            fs = (lambda: dec(key, sb)) if sb is not None else (lambda: enc(key, small))
            tracemalloc.start()
            fs()
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            out.row(algorithm=name, operation=op, us_per_op="", mb_per_s=1.0 / t.wall, cpu_s=t.cpu,
                    peak_mem_kb=f"{peak / 1024:.0f} (64 KB input)")
    out.close()


def children_cpu():
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def video():
    out = CSV("d_video", ["clip", "resolution", "seconds", "embed_s", "embed_s_per_min", "embed_cpu_s",
                          "extract_s", "extract_s_per_min", "extract_cpu_s", "peak_rss_mb"])
    for clip in available(CLIPS + [CLIP_1080]):
        src = clip_path(clip)
        info = probe(src, count_frames=True)
        secs = info.nframes / float(info.fps)
        keys, codec = film_setup(clip)
        cw = codec.encode(USER, SESSION)
        dst = WORK / f"d_{clip}.mkv"
        c0 = children_cpu()
        with Timer() as te:
            embed_ss(src, dst, cw, keys)
        c1 = children_cpu()
        with Timer() as tx:
            extract_ss(dst, src, codec)
        c2 = children_cpu()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
        out.row(clip=clip, resolution=f"{info.width}x{info.height}", seconds=secs, embed_s=te.wall,
                embed_s_per_min=te.wall * 60 / secs, embed_cpu_s=te.cpu + c1 - c0, extract_s=tx.wall,
                extract_s_per_min=tx.wall * 60 / secs, extract_cpu_s=tx.cpu + c2 - c1, peak_rss_mb=rss)
        os.remove(dst)
    out.close()


if __name__ == "__main__":
    crypto()
    video()
