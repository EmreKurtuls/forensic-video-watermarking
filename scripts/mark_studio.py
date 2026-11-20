import argparse
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from skimage.metrics import structural_similarity

from videowm.embed.ss_nonblind import DEFAULT_ALPHA, embed_video
from videowm.keys import derive_film_keys, generate_master_key
from videowm.metrics import psnr
from videowm.payload import Algorithm, PayloadCodec, pack_block
from videowm.video_io import probe, read_frames

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "presentation" / "mark_studio"
CLIPS = ROOT / "data" / "clips"
WORK = ROOT / "data" / "out" / "studio"
UPLOADS = WORK / "uploads"
MASTER = generate_master_key()
JOBS: dict[str, dict] = {}
ALGS = {"PRESENT-128": Algorithm.PRESENT128, "FF1": Algorithm.FF1}
VIDEO_EXT = {".mkv", ".mp4", ".avi", ".mov", ".webm", ".y4m"}
MAX_UPLOAD = 4 * 1024 ** 3
STATIC = {"index.html": "text/html; charset=utf-8", "tokens.css": "text/css", "base.css": "text/css"}
OUT_FILES = {"original.png": "image/png", "marked.png": "image/png", "diff20.png": "image/png",
             "original.mp4": "video/mp4", "marked.mp4": "video/mp4"}


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", *map(str, args)], check=True)


def duration(p: Path) -> float | None:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True).stdout.strip()
    try:
        return round(float(out), 1)
    except ValueError:
        return None


def film_list():
    films = []
    for folder, kind in ((CLIPS, "test clip"), (UPLOADS, "uploaded")):
        for p in sorted(folder.glob("*")) if folder.exists() else []:
            if p.suffix.lower() not in VIDEO_EXT:
                continue
            try:
                info = probe(p, count_frames=False)
                films.append({"id": f"{kind}:{p.name}", "name": p.stem, "kind": kind, "width": info.width,
                              "height": info.height, "fps": round(float(info.fps), 3), "seconds": duration(p)})
            except Exception:
                continue
    return films


def film_path(film_id: str) -> Path:
    kind, _, name = film_id.partition(":")
    folder = {"test clip": CLIPS, "uploaded": UPLOADS}.get(kind)
    p = (folder / name) if folder else None
    if p is None or p.parent != folder or not p.exists():
        raise ValueError("unknown film")
    return p


def run_job(job_id: str, req: dict):
    job = JOBS[job_id]
    out = WORK / job_id
    out.mkdir(parents=True, exist_ok=True)

    def stage(name, frac=None):
        job.update(stage=name, progress=frac)

    try:
        src_full = film_path(req["film"])
        alg = ALGS[req["alg"]]
        user, session = int(req["user"]), int(req["session"])
        seconds = max(5, min(60, int(req.get("seconds", 10))))
        film_id = re.sub(r"[^A-Za-z0-9_.-]", "_", src_full.stem)[:40].encode()

        stage("Cutting the film", None)
        src = out / "source.mkv"
        info0 = probe(src_full)
        vf = "scale='min(1920,iw)':-2:flags=bicubic" if info0.width > 1920 else "null"
        ff("-i", src_full, "-t", seconds, "-map", "0:v:0", "-an", "-vf", vf, "-c:v", "ffv1", "-level", "3",
           "-pix_fmt", "yuv420p", src)
        info = probe(src, count_frames=True)

        stage("Encrypting the viewer's ID", None)
        keys = derive_film_keys(MASTER, film_id)
        codec = PayloadCodec(keys, alg)
        block = pack_block(user, session)
        t0 = time.perf_counter()
        for _ in range(20):
            cw_bytes = codec.encode_bytes(user, session)
        enc_us = (time.perf_counter() - t0) / 20 * 1e6
        cw = codec.encode(user, session)

        stage("Embedding", 0.0)
        marked = out / "marked.mkv"
        total = info.nframes or 1
        t0 = time.perf_counter()
        embed_video(src, marked, cw, keys.stego_key, progress=lambda n: job.update(progress=n / total))
        embed_s = time.perf_counter() - t0

        stage("Measuring quality", None)
        ps, ss, maxc, mid = [], [], 0, total // 2
        for i, (a, b) in enumerate(zip(read_frames(src), read_frames(marked))):
            ps.append(psnr(a.y, b.y))
            maxc = max(maxc, int(np.max(np.abs(a.y.astype(int) - b.y.astype(int)))))
            if i % 6 == 0:
                ss.append(structural_similarity(a.y, b.y, data_range=255))
            if i == mid:
                diff = np.clip(128 + 20 * (b.y.astype(int) - a.y.astype(int)), 0, 255).astype(np.uint8)
                h, w = diff.shape
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}",
                                "-i", "-", str(out / "diff20.png")], input=diff.tobytes(), check=True)

        stage("Making previews", None)
        for name, path in (("original", src), ("marked", marked)):
            ff("-i", path, "-vf", f"select=eq(n\\,{mid})", "-frames:v", "1", out / f"{name}.png")
            ff("-i", path, "-vf", "scale='min(1280,iw)':-2", "-c:v", "libx264", "-crf", "16", "-preset", "fast",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", out / f"{name}.mp4")
        src.unlink()
        marked.unlink()

        job.update(stage="done", progress=1.0, result={
            "film": src_full.stem, "film_id": film_id.decode(), "alg": req["alg"], "user": user, "session": session,
            "width": info.width, "height": info.height, "fps": round(float(Fraction(info.fps)), 3), "frames": total,
            "seconds": round(total / float(info.fps), 2), "block": f"{block:016x}", "ciphertext": cw_bytes[:8].hex(),
            "tag": cw_bytes[8:12].hex(), "codeword": "".join(map(str, cw)), "encode_us": round(enc_us, 1),
            "embed_s": round(embed_s, 1), "psnr_y": round(float(np.mean(ps)), 2), "ssim_y": round(float(np.mean(ss)), 4),
            "max_change": maxc, "alpha": round(DEFAULT_ALPHA, 2), "still_frame": mid,
            "blocks": (info.width // 8) * (info.height // 8),
        })
    except Exception as e:
        job.update(stage="error", error=f"{type(e).__name__}: {e}")


class Handler(BaseHTTPRequestHandler):
    server_version = "MarkStudio/1"

    def log_message(self, fmt, *args):
        sys.stderr.write("  " + fmt % args + "\n")

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, ctype: str):
        size = path.stat().st_size
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        m = re.match(r"bytes=(\d*)-(\d*)$", rng or "")
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self.send_file(PAGE / "index.html", STATIC["index.html"])
        if u.path.lstrip("/") in STATIC:
            return self.send_file(PAGE / u.path.lstrip("/"), STATIC[u.path.lstrip("/")])
        if u.path == "/api/films":
            return self.send_json(film_list())
        if u.path == "/api/job":
            job = JOBS.get(parse_qs(u.query).get("id", [""])[0])
            return self.send_json(job if job else {"stage": "error", "error": "unknown job"}, 200 if job else 404)
        m = re.match(r"^/out/([0-9a-f]{32})/([a-z0-9]+\.(?:png|mp4))$", u.path)
        if m and m.group(1) in JOBS and m.group(2) in OUT_FILES and (WORK / m.group(1) / m.group(2)).exists():
            return self.send_file(WORK / m.group(1) / m.group(2), OUT_FILES[m.group(2)])
        self.send_error(404)

    def do_POST(self):
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        if u.path == "/api/mark":
            try:
                req = json.loads(self.rfile.read(length) or b"{}")
                film_path(req["film"])
                if req.get("alg") not in ALGS:
                    raise ValueError("unknown cipher")
                if not (0 <= int(req["user"]) < 2 ** 32 and 0 <= int(req["session"]) < 2 ** 24):
                    raise ValueError("user must fit in 32 bits and session in 24 bits")
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                return self.send_json({"error": str(e)}, 400)
            job_id = uuid.uuid4().hex
            JOBS[job_id] = {"id": job_id, "stage": "queued", "progress": None}
            threading.Thread(target=run_job, args=(job_id, req), daemon=True).start()
            return self.send_json({"id": job_id})
        if u.path == "/api/upload":
            name = Path(parse_qs(u.query).get("name", ["film.mp4"])[0]).name
            stem, ext = Path(name).stem, Path(name).suffix.lower()
            if ext not in VIDEO_EXT or length <= 0 or length > MAX_UPLOAD:
                return self.send_json({"error": "a video file (mkv, mp4, avi, mov, webm, y4m) up to 4 GB"}, 400)
            UPLOADS.mkdir(parents=True, exist_ok=True)
            dst = UPLOADS / (re.sub(r"[^A-Za-z0-9_.-]", "_", stem)[:60] + ext)
            with open(dst, "wb") as f:
                left = length
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            try:
                probe(dst)
            except Exception:
                dst.unlink(missing_ok=True)
                return self.send_json({"error": "FFmpeg cannot read this file"}, 400)
            return self.send_json({"id": f"uploaded:{dst.name}"})
        self.send_error(404)


def main():
    ap = argparse.ArgumentParser(description='Mark studio: pick a film and a cipher, mark it with the project code, compare the film before and after.')
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    if shutil.which("ffmpeg") is None:
        sys.exit("FFmpeg is required")
    WORK.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"Mark studio: http://127.0.0.1:{a.port}  (Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
