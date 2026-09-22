"""Disk-backed exact and near-duplicate filtering for large corpora."""

from __future__ import annotations

import hashlib
import re
import sqlite3
import unicodedata
from pathlib import Path


TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def normalize_for_dedup(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(TOKEN_RE.findall(text))


def exact_text_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_dedup(text).encode("utf-8")).hexdigest()


def simhash64(text: str, shingle_size: int = 3) -> tuple[int, int]:
    """Return a deterministic 64-bit SimHash and normalized token count."""

    tokens = TOKEN_RE.findall(normalize_for_dedup(text))
    if not tokens:
        return 0, 0
    if len(tokens) < shingle_size:
        shingles = ["\0".join(tokens)]
    else:
        shingles = [
            "\0".join(tokens[index : index + shingle_size])
            for index in range(len(tokens) - shingle_size + 1)
        ]
    weights = [0] * 64
    for shingle in shingles:
        value = int.from_bytes(
            hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big"
        )
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    signature = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            signature |= 1 << bit
    return signature, len(tokens)


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class SQLiteDeduplicator:
    """Bound Python memory by keeping exact hashes and SimHash bands in SQLite."""

    def __init__(
        self,
        path: str | Path,
        *,
        near_hamming_threshold: int = 3,
        minimum_near_tokens: int = 20,
        commit_interval: int = 1000,
    ):
        if near_hamming_threshold < 0 or near_hamming_threshold > 64:
            raise ValueError("near_hamming_threshold must be between 0 and 64")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.near_hamming_threshold = near_hamming_threshold
        self.minimum_near_tokens = minimum_near_tokens
        self.commit_interval = max(int(commit_interval), 1)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS exact_hashes (
                text_hash TEXT PRIMARY KEY,
                session_id TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS near_hashes (
                band INTEGER NOT NULL,
                band_value INTEGER NOT NULL,
                signature TEXT NOT NULL,
                session_id TEXT NOT NULL,
                PRIMARY KEY (band, band_value, signature, session_id)
            );
            CREATE INDEX IF NOT EXISTS near_lookup
                ON near_hashes (band, band_value);
            """
        )
        self._pending = 0

    @staticmethod
    def _bands(signature: int) -> tuple[tuple[int, int], ...]:
        return tuple(
            (band, (signature >> (band * 16)) & 0xFFFF) for band in range(4)
        )

    def classify_and_add(self, text: str, session_id: str) -> str | None:
        """Return a rejection reason, or add the accepted text and return None."""

        text_hash = exact_text_hash(text)
        exact_match = self.connection.execute(
            "SELECT session_id FROM exact_hashes WHERE text_hash = ?", (text_hash,)
        ).fetchone()
        if exact_match is not None:
            return "exact_duplicate"

        signature, token_count = simhash64(text)
        if token_count >= self.minimum_near_tokens:
            candidates: dict[str, str] = {}
            for band, band_value in self._bands(signature):
                for candidate_signature, candidate_id in self.connection.execute(
                    "SELECT signature, session_id FROM near_hashes "
                    "WHERE band = ? AND band_value = ?",
                    (band, band_value),
                ):
                    candidates[candidate_signature] = candidate_id
            if any(
                hamming_distance(signature, int(candidate, 16))
                <= self.near_hamming_threshold
                for candidate in candidates
            ):
                return "near_duplicate"

        self.connection.execute(
            "INSERT INTO exact_hashes(text_hash, session_id) VALUES (?, ?)",
            (text_hash, session_id),
        )
        if token_count >= self.minimum_near_tokens:
            signature_hex = f"{signature:016x}"
            self.connection.executemany(
                "INSERT INTO near_hashes(band, band_value, signature, session_id) "
                "VALUES (?, ?, ?, ?)",
                [
                    (band, band_value, signature_hex, session_id)
                    for band, band_value in self._bands(signature)
                ],
            )
        self._pending += 1
        if self._pending >= self.commit_interval:
            self.connection.commit()
            self._pending = 0
        return None

    def close(self) -> None:
        if self.connection is not None:
            self.connection.commit()
            self.connection.close()
            self.connection = None

    def __enter__(self) -> "SQLiteDeduplicator":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
