import argparse
import sys
import time

from .db import PlatformDB
from .embed.ss_nonblind import DEFAULT_ALPHA, embed_video
from .extract import identify
from .keys import derive_film_keys, generate_master_key, load_master_key, save_master_key
from .payload import Algorithm, PayloadCodec

ALGS = {"present": Algorithm.PRESENT128, "ff1": Algorithm.FF1}


def cmd_keygen(a) -> int:
    try:
        save_master_key(a.key, generate_master_key(), overwrite=a.force)
    except FileExistsError:
        print(f"error: {a.key} exists (use --force to replace it; marked copies made with it become untraceable)",
              file=sys.stderr)
        return 1
    print(f"master key written to {a.key} (mode 0600)")
    return 0


def cmd_register(a) -> int:
    db = PlatformDB(a.db)
    try:
        db.add_film(a.film_id, a.original, ALGS[a.alg], a.alpha)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"film {a.film_id} registered: algorithm {ALGS[a.alg].name}, alpha {a.alpha:.2f}")
    return 0


def _film_codec(db, a):
    film = db.get_film(a.film_id)
    keys = derive_film_keys(load_master_key(a.key), film.film_id.encode())
    return film, keys, PayloadCodec(keys, film.algorithm)


def cmd_embed(a) -> int:
    db = PlatformDB(a.db)
    film, keys, codec = _film_codec(db, a)
    session = db.next_session(a.user_id)
    t = time.perf_counter()
    stats = embed_video(film.original, a.out, codec.encode(a.user_id, session), keys.stego_key, film.alpha)
    db.record_issue(film.film_id, a.user_id, session, film.algorithm, a.out)
    print(f"copy for user {a.user_id} (session {session}) written to {a.out}: {stats.frames} frames, "
          f"PSNR(Y) {stats.psnr_y:.1f} dB, {time.perf_counter() - t:.1f} s")
    return 0


def cmd_trace(a) -> int:
    db = PlatformDB(a.db)
    film, keys, codec = _film_codec(db, a)
    t = time.perf_counter()
    ident, ex = identify(a.pirate, film.original, codec)
    took = time.perf_counter() - t
    if ident is None:
        print(f"no valid watermark found ({ex.pirate_frames} frames analysed, {took:.1f} s)")
        return 2
    issue = db.find_issue(film.film_id, ident.user_id, ident.session)
    print(f"watermark valid: user {ident.user_id}, session {ident.session} "
          f"({ident.corrected_bits} bit errors corrected, {ex.pirate_frames} frames, {took:.1f} s)")
    if issue is None:
        print("warning: MAC is valid but no issued copy matches; check the database")
        return 3
    print(f"issued {issue.issued_at} as {issue.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="videowm", description="Forensic watermarking prototype for a film platform")
    sub = p.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keygen", help="create the platform master key (CSPRNG)")
    k.add_argument("--key", required=True)
    k.add_argument("--force", action="store_true")
    k.set_defaults(func=cmd_keygen)
    r = sub.add_parser("register-film", help="register an unmarked master copy")
    r.add_argument("--db", required=True)
    r.add_argument("--film-id", required=True)
    r.add_argument("--original", required=True)
    r.add_argument("--alg", choices=sorted(ALGS), default="present")
    r.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    r.set_defaults(func=cmd_register)
    for name, func, helptext in (("embed", cmd_embed, "make a marked copy for one user"),
                                 ("trace", cmd_trace, "find who a pirate copy was issued to")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--db", required=True)
        s.add_argument("--key", required=True)
        s.add_argument("--film-id", required=True)
        s.set_defaults(func=func)
        if name == "embed":
            s.add_argument("--user-id", type=int, required=True)
            s.add_argument("--out", required=True)
        else:
            s.add_argument("--pirate", required=True)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except (KeyError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
