# Forensic video watermarking for a film streaming platform

COM0430 Cryptography, Phase II (2026–2027 Fall).

Every viewing session gets its own copy of the film. The viewer's ID and session number are encrypted, authenticated and hidden invisibly in the picture. When a copy appears on a pirate site, the platform reads the mark back and identifies the viewer who leaked it. The mark survives re-encoding, downscaling, trimming and packet loss.

## How it works

| Part | Choice |
|---|---|
| Payload | 64-bit block `VER 4 \| user_id 32 \| session 24 \| reserved 4` |
| Ciphers | **PRESENT-128** (ISO/IEC 29192-2) and **FF1** (NIST SP 800-38G, film ID as tweak, AES-128 inside). Both keep the ciphertext at 64 bits. |
| Integrity | SipHash-2-4 truncated to 32 bits, encrypt-then-MAC with a key commitment. The tag is checked in constant time before decrypting. |
| Keys | Master key from the OS CSPRNG (`secrets`). Per-film keys come from HKDF-SHA256, and the embedding patterns from SHAKE-256. |
| Error correction | BCH(188,96), corrects 12 bit errors. 188 bits are embedded. |
| Hiding | Non-blind additive spread spectrum in the Y channel: low 8×8 DCT band (zigzag 1–5), texture masking, bits held for key-dependent groups of about 1 s, soft combining. |
| Extraction | Rescale the pirate copy, align it to the original frame by frame (dynamic programming), correlate, then BCH → MAC → decrypt. A wrong key, damage or a forgery all give the same `NoValidMark`. |
| Baselines | Blind QIM/STDM and LSB matching, used for comparison only. |

All cryptography is implemented in pure Python and checked against official known-answer tests: FIPS 197, SP 800-38A and SP 800-38G for AES and FF1, the SipHash reference vectors, and OpenTitan vectors for PRESENT-128. That keeps the speed comparison of the two ciphers fair. `cryptography` is used only for HKDF and as a cross-check in the tests.

## Setup

Requires Python ≥ 3.11 and FFmpeg (with `libvmaf` for the VMAF metric).

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m pytest -q
```

## Usage

```bash
export PYTHONPATH=src
PY=./.venv/bin/python
$PY -m videowm keygen        --key platform.key
$PY -m videowm register-film --db platform.sqlite --film-id BBB-0001 --original film.mp4 --alg present   # or --alg ff1
$PY -m videowm embed         --db platform.sqlite --key platform.key --film-id BBB-0001 --user-id 1001 --out copy.mkv
$PY -m videowm trace         --db platform.sqlite --key platform.key --film-id BBB-0001 --pirate leak.mp4
```

The master key is written to its own file with mode 0600. The SQLite database holds films, sessions and issued copies, but no keys. Never commit `*.key` or `*.sqlite` files.

## Test data

Videos are not in the repository. Download them into `data/raw/`:

```bash
mkdir -p data/raw/bbb720 data/raw/sintel720 data/raw/tos720 && cd data/raw
curl -o foreman_cif.y4m https://media.xiph.org/video/derf/y4m/foreman_cif.y4m
curl -o akiyo_cif.y4m   https://media.xiph.org/video/derf/y4m/akiyo_cif.y4m
sh ../download.sh            # Big Buck Bunny, Sintel, Tears of Steel (720p), park_joy 720p50, crowd_run 1080p50
cd ../..
PYTHONPATH=src ./.venv/bin/python scripts/prepare_dataset.py   # cuts the 10 s test clips into data/clips/
```

`data/resume_xiph.sh` resumes the large Xiph downloads if they stall.

### Sources and credits

No video file is redistributed in this repository. The only material taken from the videos is a few still frames in `presentation/`, listed below.

| Clip | Source | Licence / terms |
|---|---|---|
| Big Buck Bunny | © 2008 Blender Foundation, [peach.blender.org](https://peach.blender.org) | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). The frames in `presentation/network_sim/` are taken from it and modified (watermark added). |
| Sintel | © Blender Foundation, [durian.blender.org](https://durian.blender.org) | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) |
| Tears of Steel | (CC) Blender Foundation, [mango.blender.org](https://mango.blender.org) | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). The frames in `presentation/watermark_animation/` are taken from it and modified (watermark added). |
| akiyo, foreman (CIF) | Standard video test sequences, distributed by [Xiph.org Video Test Media](https://media.xiph.org/video/derf/) | Used for research and testing only, under the terms stated there |
| park_joy, crowd_run | SVT (Sveriges Television) test sequences, distributed by [Xiph.org Video Test Media](https://media.xiph.org/video/derf/) | Used for research and testing only, under the terms stated there |

## Evaluation

The scripts in `evaluation/` implement the test scenarios of the assignment:

| Script | Scenario |
|---|---|
| `a_accuracy.py` | (a) accuracy, small and large data, MP4 / MKV / AVI |
| `b_degradation.py` | (b) H.264/H.265 re-encoding, scaling, trimming and frame drops, MPEG-TS packet loss |
| `c_wrong_key.py` | (c) wrong key, wrong film, unmarked and forged copies |
| `d_performance.py` | (d) speed, CPU and memory of PRESENT-128 vs FF1, embedding and extraction cost |
| `e_quality.py` | (e) PSNR, SSIM, VMAF, BER, encryption and embedding times |
| `f_capacity.py` | (f) raw and usable capacity |
| `g_histogram.py` | (g) colour histograms and steganalysis |

`sh evaluation/run_all.sh` runs all of them, which takes hours. Logs go to `data/out/logs/`; CSV files and figures go to `results/`, where the measured results are already included.

For a quick run on one clip:

```bash
VIDEOWM_CLIPS=akiyo_cif VIDEOWM_RESULTS=data/out/smoke_results PYTHONPATH=src ./.venv/bin/python evaluation/e_quality.py
```

## Repository layout

| Path | Content |
|---|---|
| `src/videowm/` | The package: `ciphers/` (AES, FF1, PRESENT, SipHash), `keys.py`, `payload.py`, `video_io.py`, `dct.py`, `embed/` (spread spectrum, QIM, LSB), `align.py`, `extract.py`, `db.py`, `cli.py`, `attacks.py`, `metrics.py`, `bulk.py` |
| `tests/` | pytest unit tests, including the cipher KATs |
| `evaluation/` | Test scenarios (a)–(g) and plots |
| `results/` | Measured results (CSV) and figures |
| `scripts/` | Dataset preparation, comparison videos, data for the demo pages, viewing-test clips |
| `presentation/` | HTML demo pages: watermark walkthrough, network simulation, Mark studio |
| `subjective_test/` | 2AFC viewing-test page (generate the clips with `scripts/make_2afc_clips.py`) |
| `experiments/` | Prototype scripts behind the design decisions (separate `requirements.txt`) |

### Demo pages

- `presentation/watermark_animation/index.html` and `presentation/network_sim/index.html` open directly in a browser. To regenerate their data, run `scripts/make_animation_data.py` (followed by `presentation/watermark_animation/build.py`) and `scripts/make_network_sim_data.py`.
- Mark studio runs the real pipeline from a local web page. Start it with `PYTHONPATH=src ./.venv/bin/python scripts/mark_studio.py`, then open http://127.0.0.1:8765.
