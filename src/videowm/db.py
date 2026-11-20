import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from .payload import Algorithm

SCHEMA = """
CREATE TABLE IF NOT EXISTS films (
    film_id    TEXT PRIMARY KEY,
    original   TEXT NOT NULL,
    algorithm  INTEGER NOT NULL,
    alpha      REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS user_sessions (
    user_id      INTEGER PRIMARY KEY,
    next_session INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS issued (
    film_id   TEXT NOT NULL REFERENCES films(film_id),
    user_id   INTEGER NOT NULL,
    session   INTEGER NOT NULL,
    algorithm INTEGER NOT NULL,
    output    TEXT NOT NULL,
    issued_at TEXT NOT NULL,
    PRIMARY KEY (film_id, user_id, session)
);
"""
MAX_SESSION = 2 ** 24


@dataclass(frozen=True)
class Film:
    film_id: str
    original: str
    algorithm: Algorithm
    alpha: float


@dataclass(frozen=True)
class Issue:
    film_id: str
    user_id: int
    session: int
    algorithm: Algorithm
    output: str
    issued_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PlatformDB:
    def __init__(self, path):
        self.conn = sqlite3.connect(str(path), isolation_level=None)
        self.conn.executescript(SCHEMA)

    def close(self):
        self.conn.close()

    def add_film(self, film_id: str, original: str, algorithm: Algorithm, alpha: float) -> None:
        try:
            self.conn.execute("INSERT INTO films VALUES (?, ?, ?, ?, ?)",
                              (film_id, original, int(algorithm), float(alpha), _now()))
        except sqlite3.IntegrityError:
            raise ValueError(f"film {film_id!r} is already registered") from None

    def get_film(self, film_id: str) -> Film:
        row = self.conn.execute("SELECT film_id, original, algorithm, alpha FROM films WHERE film_id = ?",
                                (film_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown film {film_id!r}")
        return Film(row[0], row[1], Algorithm(row[2]), row[3])

    def next_session(self, user_id: int) -> int:
        if not 0 <= user_id < 2 ** 32:
            raise ValueError("user_id must fit in 32 bits")
        c = self.conn
        c.execute("BEGIN IMMEDIATE")
        try:
            row = c.execute("SELECT next_session FROM user_sessions WHERE user_id = ?", (user_id,)).fetchone()
            s = 0 if row is None else row[0]
            if s >= MAX_SESSION:
                raise ValueError("session counter exhausted for this user")
            c.execute("INSERT INTO user_sessions VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE SET next_session = ?",
                      (user_id, s + 1, s + 1))
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise
        return s

    def record_issue(self, film_id: str, user_id: int, session: int, algorithm: Algorithm, output: str) -> None:
        self.conn.execute("INSERT INTO issued VALUES (?, ?, ?, ?, ?, ?)",
                          (film_id, user_id, session, int(algorithm), output, _now()))

    def find_issue(self, film_id: str, user_id: int, session: int) -> Issue | None:
        row = self.conn.execute("SELECT * FROM issued WHERE film_id = ? AND user_id = ? AND session = ?",
                                (film_id, user_id, session)).fetchone()
        return None if row is None else Issue(row[0], row[1], row[2], Algorithm(row[3]), row[4], row[5])
