import json
import secrets
import subprocess
from fractions import Fraction
from pathlib import Path

from videowm.embed.ss_nonblind import embed_video
from videowm.keys import derive_film_keys, generate_master_key
from videowm.payload import Algorithm, PayloadCodec
from videowm.video_io import probe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "subjective_test" / "clips"
WORK = ROOT / "data" / "out" / "2afc"
CLIPS = ["akiyo_cif", "foreman_cif", "park_joy_720p", "bbb_action_720p", "sintel_dark_720p", "tos_static_720p"]
SECONDS = 5


def encode(src, dst, start, frames):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf",
                    f"trim=start_frame={start}:end_frame={start + frames},setpts=PTS-STARTPTS", "-an", "-c:v", "libx264",
                    "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)], check=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in list(OUT.glob("*.mp4")) + list(OUT.glob("manifest.js")):
        old.unlink()
    WORK.mkdir(parents=True, exist_ok=True)
    master = generate_master_key()
    manifest = []
    for clip in CLIPS:
        src = ROOT / "data" / "clips" / f"{clip}.mkv"
        info = probe(src)
        frames = round(SECONDS * float(Fraction(info.fps)))
        keys = derive_film_keys(master, clip.encode())
        marked = WORK / f"{clip}_marked.mkv"
        embed_video(src, marked, PayloadCodec(keys, Algorithm.PRESENT128).encode(1001, 0), keys.stego_key)
        for part in (0, 1):
            names = {}
            for kind, path in (("original", src), ("marked", marked)):
                names[kind] = f"{secrets.token_hex(6)}.mp4"
                encode(path, OUT / names[kind], part * frames, frames)
            manifest.append({"clip": clip, "part": part, "width": info.width, "height": info.height,
                             "fps": float(Fraction(info.fps)), "original": names["original"], "marked": names["marked"]})
            print(clip, part, names, flush=True)
        marked.unlink()
    (OUT / "manifest.js").write_text("window.CLIPS = " + json.dumps(manifest) + ";\n")
    print("wrote", OUT / "manifest.js")


if __name__ == "__main__":
    main()
