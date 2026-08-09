#!/usr/bin/env python3
"""Measure stored-top-move route depth for Arrow Duel positions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, local
from typing import Sequence

from presolve_lichess_puzzles import Evaluation, StockfishEngine, StockfishError


SCHEMA_VERSION = 4
ROUTE_MULTIPV = 2
COMPACT_ENCODING_VERSION = 1
COMPACT_CHUNK_SIZE = 4096
COMPACT_MAX_UNIQUE_PLIES = 14
PACK_FORMAT_BINARY_V1 = (
    1,
    "chessticize-core-pack",
    2,
    "chessticize-position",
    1,
    "chessticize-uci16",
    1,
)
PACK_PIECES = (None, "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")
PACK_PROMOTIONS = (None, "q", "r", "b", "n")


@dataclass(frozen=True)
class RouteConfig:
    multipv: int = 2
    eval_tolerance_cp: int = 100
    max_unique_plies: int = 10

    def __post_init__(self) -> None:
        if self.multipv != ROUTE_MULTIPV:
            raise ValueError(f"multipv must be {ROUTE_MULTIPV}")
        if self.eval_tolerance_cp < 0:
            raise ValueError("eval_tolerance_cp must not be negative")
        if self.max_unique_plies < 2:
            raise ValueError("max_unique_plies must be at least 2")
        if self.max_unique_plies > COMPACT_MAX_UNIQUE_PLIES:
            raise ValueError(
                "max_unique_plies must not exceed "
                f"{COMPACT_MAX_UNIQUE_PLIES} for the four-bit encoding"
            )


@dataclass(frozen=True)
class PuzzleInput:
    puzzle_id: str
    initial_fen: str
    rating: int
    stockfish_bestmove: str


@dataclass(frozen=True)
class PackLayout:
    puzzle_count: int
    row_encoding: str


@dataclass(frozen=True)
class RouteMetric:
    puzzle_id: str
    top_move_unique_plies: int


@dataclass(frozen=True)
class OptionSet:
    lines: tuple[Evaluation, ...]
    good_lines: tuple[Evaluation, ...]


_worker_state = local()
_worker_engines: list[StockfishEngine] = []
_worker_engines_lock = Lock()
_worker_depth = 10
_worker_config = RouteConfig()


def _initialize_worker(
    stockfish_path: str,
    threads: int,
    hash_mb: int,
    depth: int,
    config: RouteConfig,
) -> None:
    global _worker_depth, _worker_config
    engine = StockfishEngine(stockfish_path, threads, hash_mb)
    _worker_state.engine = engine
    with _worker_engines_lock:
        _worker_engines.append(engine)
    _worker_depth = depth
    _worker_config = config


def _close_worker_engines() -> None:
    with _worker_engines_lock:
        engines = list(_worker_engines)
        _worker_engines.clear()
    for engine in engines:
        engine.close()


def _analyze_batch(puzzles: Sequence[PuzzleInput]) -> tuple[RouteMetric, ...]:
    engine = getattr(_worker_state, "engine", None)
    if engine is None:
        raise RuntimeError("route-metric worker has no Stockfish engine")
    results = []
    for puzzle in puzzles:
        engine.start_new_game()
        results.append(
            analyze_route(
                engine=engine,
                puzzle=puzzle,
                depth=_worker_depth,
                config=_worker_config,
            )
        )
    return tuple(results)


def analyze_route(
    *,
    engine: StockfishEngine,
    puzzle: PuzzleInput,
    depth: int,
    config: RouteConfig,
) -> RouteMetric:
    path: list[str] = []
    candidate = normalize_uci(puzzle.stockfish_bestmove)
    line = engine.evaluate(
        puzzle.initial_fen,
        depth,
        searchmoves=(candidate,),
    )
    while True:
        variation = tuple(map(normalize_uci, line.principal_variation))
        if not variation:
            raise StockfishError(
                f"Puzzle {puzzle.puzzle_id!r} returned an empty principal variation"
            )
        path.append(variation[0])
        if len(variation) == 1:
            return RouteMetric(puzzle.puzzle_id, len(path))
        if len(path) >= config.max_unique_plies:
            return RouteMetric(
                puzzle.puzzle_id,
                config.max_unique_plies + 1,
            )

        path.append(variation[1])
        if len(variation) == 2:
            return RouteMetric(puzzle.puzzle_id, len(path))
        if len(path) >= config.max_unique_plies:
            return RouteMetric(
                puzzle.puzzle_id,
                config.max_unique_plies + 1,
            )

        options = evaluate_options(
            engine,
            puzzle.initial_fen,
            path,
            depth=depth,
            config=config,
        )
        if len(options.good_lines) != 1:
            return RouteMetric(puzzle.puzzle_id, len(path))
        line = options.good_lines[0]


def evaluate_options(
    engine: StockfishEngine,
    fen: str,
    moves: Sequence[str],
    *,
    depth: int,
    config: RouteConfig,
) -> OptionSet:
    lines = engine.evaluate_multipv(
        fen,
        depth,
        multipv=config.multipv,
        moves=moves,
    )
    best = lines[0]
    good_lines = tuple(
        line
        for line in lines
        if is_competitive(
            best,
            line,
            fen=fen,
            tolerance_cp=config.eval_tolerance_cp,
        )
    )
    if not good_lines:
        raise StockfishError("Stockfish's first MultiPV line was not competitive")
    return OptionSet(lines=lines, good_lines=good_lines)


def normalize_uci(move: str) -> str:
    return move.strip().lower()


def decode_pack_move(value: bytes) -> str:
    if len(value) != 2:
        raise ValueError(
            f"Invalid binary Core Pack move length {len(value)}; expected 2"
        )
    encoded = value[0] | (value[1] << 8)
    if encoded & 0x8000:
        raise ValueError("Invalid reserved Core Pack move bit")
    from_square = encoded & 0x3F
    to_square = (encoded >> 6) & 0x3F
    if from_square == to_square:
        raise ValueError("Encoded Core Pack move must use distinct squares")
    promotion_code = (encoded >> 12) & 0x07
    if promotion_code >= len(PACK_PROMOTIONS):
        raise ValueError(f"Invalid Core Pack promotion code {promotion_code}")
    promotion = PACK_PROMOTIONS[promotion_code] or ""
    if promotion:
        from_rank = (from_square >> 3) + 1
        to_rank = (to_square >> 3) + 1
        if (from_rank, to_rank) not in {(7, 8), (2, 1)}:
            raise ValueError("Invalid Core Pack promotion rank transition")
    return f"{pack_square_name(from_square)}{pack_square_name(to_square)}{promotion}"


def decode_pack_position(value: bytes) -> str:
    if len(value) < 10:
        raise ValueError(
            f"Invalid binary Core Pack position length {len(value)}; expected at least 10"
        )
    occupied_count = sum(byte.bit_count() for byte in value[:8])
    expected_length = 10 + (occupied_count + 1) // 2
    if len(value) != expected_length:
        raise ValueError(
            f"Invalid binary Core Pack position length {len(value)}; "
            f"expected {expected_length}"
        )
    metadata = value[8] | (value[9] << 8)
    if metadata & 0xF000:
        raise ValueError("Invalid reserved Core Pack position bits")
    en_passant_code = (metadata >> 5) & 0x7F
    if en_passant_code > 64:
        raise ValueError(f"Invalid Core Pack en-passant code {en_passant_code}")

    pieces: list[str | None] = [None] * 64
    piece_index = 0
    for square in range(64):
        if not value[square >> 3] & (1 << (square & 7)):
            continue
        encoded_piece = value[10 + (piece_index >> 1)]
        code = encoded_piece & 0x0F if piece_index % 2 == 0 else encoded_piece >> 4
        if code >= len(PACK_PIECES) or PACK_PIECES[code] is None:
            raise ValueError(f"Invalid Core Pack piece code {code}")
        pieces[square] = PACK_PIECES[code]
        piece_index += 1
    if occupied_count % 2 and value[-1] >> 4:
        raise ValueError("Invalid Core Pack position padding nibble")

    ranks = []
    for rank in range(7, -1, -1):
        encoded_rank = ""
        empty_count = 0
        for file_index in range(8):
            piece = pieces[rank * 8 + file_index]
            if piece is None:
                empty_count += 1
                continue
            if empty_count:
                encoded_rank += str(empty_count)
                empty_count = 0
            encoded_rank += piece
        if empty_count:
            encoded_rank += str(empty_count)
        ranks.append(encoded_rank)
    side = "b" if metadata & 1 else "w"
    castling = "".join(
        symbol
        for bit, symbol in ((1, "K"), (2, "Q"), (3, "k"), (4, "q"))
        if metadata & (1 << bit)
    ) or "-"
    en_passant = "-" if en_passant_code == 0 else pack_square_name(en_passant_code - 1)
    return f"{'/'.join(ranks)} {side} {castling} {en_passant} 0 1"


def pack_square_name(square: int) -> str:
    return f"{chr(97 + (square & 7))}{(square >> 3) + 1}"


def decode_puzzle_row(row: Sequence[object], row_encoding: str) -> PuzzleInput:
    puzzle_id, initial_fen, rating, bestmove = row
    if not isinstance(puzzle_id, str) or not isinstance(rating, int):
        raise ValueError("Core Pack puzzle ID or rating has an invalid type")
    if row_encoding == "binary-v1":
        if not isinstance(initial_fen, bytes) or not isinstance(bestmove, bytes):
            raise ValueError("Binary Core Pack payload columns must contain BLOBs")
        return PuzzleInput(
            puzzle_id,
            decode_pack_position(initial_fen),
            rating,
            decode_pack_move(bestmove),
        )
    if not isinstance(initial_fen, str) or not isinstance(bestmove, str):
        raise ValueError("Legacy Core Pack payload columns must contain TEXT")
    fen_fields = initial_fen.strip().split()
    if len(fen_fields) == 4:
        initial_fen = f"{initial_fen.strip()} 0 1"
    elif len(fen_fields) != 6:
        raise ValueError(f"Invalid legacy Core Pack FEN: {initial_fen!r}")
    return PuzzleInput(puzzle_id, initial_fen, rating, normalize_uci(bestmove))


def side_for_initial_player(value: float, fen: str) -> float:
    parts = fen.split()
    if len(parts) < 2 or parts[1] not in {"w", "b"}:
        raise ValueError(f"Invalid FEN: {fen!r}")
    return value if parts[1] == "w" else -value


def mate_for_initial_player(mate_in: int | None, fen: str) -> int | None:
    if mate_in is None:
        return None
    return int(side_for_initial_player(mate_in, fen))


def is_competitive(
    best: Evaluation,
    alternative: Evaluation,
    *,
    fen: str,
    tolerance_cp: int,
) -> bool:
    best_mate = mate_for_initial_player(best.mate_in, fen)
    alternative_mate = mate_for_initial_player(alternative.mate_in, fen)
    if best_mate is not None:
        return alternative_mate == best_mate
    best_score = side_for_initial_player(best.score_cp, fen)
    alternative_score = side_for_initial_player(alternative.score_cp, fen)
    return best_score - alternative_score <= tolerance_cp


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_compact_metric(
    top_move_unique_plies: int | None,
) -> int:
    """Encode one Core Pack row into one four-bit Arrow Duel metric."""
    if top_move_unique_plies is None:
        return 0
    if not 1 <= top_move_unique_plies <= 15:
        raise ValueError("route-depth code must be from 1 to 15")
    return top_move_unique_plies


def decode_compact_metric(value: int) -> int | None:
    """Decode one four-bit Arrow Duel route metric."""
    if not 0 <= value <= 15:
        raise ValueError("compact metric must be one nibble")
    return value or None


def sqlite_read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def validate_pack(connection: sqlite3.Connection) -> PackLayout:
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(puzzles)").fetchall()
    }
    required = {
        "id",
        "initial_fen",
        "solution_moves",
        "rating",
        "stockfish_bestmove",
    }
    missing = sorted(required - columns)
    if missing:
        raise ValueError("Core Pack puzzles table is missing: " + ", ".join(missing))
    has_format = connection.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = 'pack_format'
        """
    ).fetchone()
    if has_format:
        formats = connection.execute(
            """
            SELECT
              id,
              format_id,
              pack_schema_version,
              position_codec,
              position_codec_version,
              move_codec,
              move_codec_version
            FROM pack_format
            ORDER BY id
            """
        ).fetchall()
        if formats != [PACK_FORMAT_BINARY_V1]:
            raise ValueError(f"Unsupported Core Pack format: {formats!r}")
        row_encoding = "binary-v1"
        expected_types = ("blob", "blob", "blob")
    else:
        row_encoding = "legacy-text"
        expected_types = ("text", "text", "text")
    actual_types = connection.execute(
        """
        SELECT
          typeof(initial_fen),
          typeof(solution_moves),
          typeof(stockfish_bestmove)
        FROM puzzles
        LIMIT 1
        """
    ).fetchone()
    if actual_types != expected_types:
        raise ValueError(
            f"Core Pack {row_encoding} columns have types {actual_types!r}; "
            f"expected {expected_types!r}"
        )
    return PackLayout(
        puzzle_count=int(
            connection.execute("SELECT COUNT(*) FROM puzzles").fetchone()[0]
        ),
        row_encoding=row_encoding,
    )


def arrow_duel_input_sql(row_encoding: str) -> str:
    if row_encoding == "binary-v1":
        return """
          length(p.stockfish_bestmove) = 2
          AND length(p.solution_moves) >= 2
          AND substr(hex(p.stockfish_bestmove), 3, 1) = '0'
          AND substr(hex(p.solution_moves), 3, 1) = '0'
          AND p.stockfish_bestmove != substr(p.solution_moves, 1, 2)
        """
    if row_encoding != "legacy-text":
        raise ValueError(f"Unsupported Core Pack row encoding: {row_encoding}")
    return """
      length(trim(p.stockfish_bestmove)) = 4
      AND length(
        CASE
          WHEN instr(trim(p.solution_moves), ' ') = 0 THEN trim(p.solution_moves)
          ELSE substr(trim(p.solution_moves), 1, instr(trim(p.solution_moves), ' ') - 1)
        END
      ) = 4
      AND trim(p.stockfish_bestmove) !=
        CASE
          WHEN instr(trim(p.solution_moves), ' ') = 0 THEN trim(p.solution_moves)
          ELSE substr(trim(p.solution_moves), 1, instr(trim(p.solution_moves), ' ') - 1)
        END
    """


def analysis_identity(
    *,
    input_path: Path,
    input_sha256: str,
    depth: int,
    stockfish_path: str,
    engine_name: str,
    engine_author: str,
    threads_per_job: int,
    hash_mb: int,
    jobs: int,
    batch_size: int,
    row_encoding: str,
    config: RouteConfig,
) -> dict[str, object]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "inputPath": str(input_path.resolve()),
        "inputSha256": f"sha256:{input_sha256}",
        "inputRowEncoding": row_encoding,
        "stockfish": {
            "path": stockfish_path,
            "name": engine_name,
            "author": engine_author,
            "depth": depth,
            "threadsPerJob": threads_per_job,
            "hashMbPerJob": hash_mb,
            "newGamePerPuzzle": True,
        },
        "jobs": jobs,
        "batchSize": batch_size,
        "routeMetrics": {
            **asdict(config),
            "censoredTopMoveUniqueCode": config.max_unique_plies + 1,
        },
    }


def initialize_output(
    path: Path,
    identity: dict[str, object],
    *,
    overwrite: bool,
    resume: bool,
) -> tuple[sqlite3.Connection, str, int]:
    if path.exists() and overwrite:
        path.unlink()
    if path.exists() and not resume:
        raise FileExistsError(
            f"Output already exists: {path}; pass --resume or --overwrite"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS arrow_duel_metric_definitions (
          id INTEGER PRIMARY KEY,
          key TEXT NOT NULL UNIQUE,
          description TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS arrow_duel_route_metrics (
          puzzle_id TEXT PRIMARY KEY,
          top_move_unique_plies INTEGER NOT NULL
            CHECK (top_move_unique_plies >= 1)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS analysis_metadata (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        ) WITHOUT ROWID;
        """
    )
    connection.executemany(
        """
        INSERT OR IGNORE INTO arrow_duel_metric_definitions
          (id, key, description)
        VALUES (?, ?, ?)
        """,
        [
            (
                1,
                "top_move_unique_plies",
                "top-move plies until branching or terminal; max+1 is censored",
            ),
            (
                2,
                "unique_followup_moves",
                "derived forced puzzle-side moves after the displayed stored "
                "top move; paired ply depths share a count and the final "
                "bucket is censored",
            ),
        ],
    )
    identity_json = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    stored = connection.execute(
        "SELECT value FROM analysis_metadata WHERE key = 'identity'"
    ).fetchone()
    if stored is None:
        connection.executemany(
            "INSERT INTO analysis_metadata (key, value) VALUES (?, ?)",
            [
                ("identity", identity_json),
                ("last_puzzle_id", ""),
                ("processed_count", "0"),
            ],
        )
        connection.commit()
    elif stored[0] != identity_json:
        raise ValueError("Existing output identity does not match this route run")
    last_id = connection.execute(
        "SELECT value FROM analysis_metadata WHERE key = 'last_puzzle_id'"
    ).fetchone()[0]
    processed = int(
        connection.execute(
            "SELECT value FROM analysis_metadata WHERE key = 'processed_count'"
        ).fetchone()[0]
    )
    return connection, last_id, processed


def chunks(
    values: Sequence[PuzzleInput], size: int
) -> list[Sequence[PuzzleInput]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def build_compact_output(
    *,
    input_path: Path,
    analysis_path: Path,
    compact_path: Path,
    identity: dict[str, object],
    config: RouteConfig,
    replace_existing: bool,
) -> dict[str, object]:
    """Write rowid-addressed, four-bit metrics without duplicating puzzle IDs."""
    actual_input_sha256 = f"sha256:{sha256(input_path)}"
    if actual_input_sha256 != identity["inputSha256"]:
        raise ValueError(
            "Compact output input SHA-256 does not match the analyzed Core Pack"
        )
    if compact_path.exists():
        if not replace_existing:
            raise FileExistsError(
                f"Compact output already exists: {compact_path}; pass --overwrite "
                "or --resume"
            )
        compact_path.unlink()
    compact_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(compact_path)
    try:
        connection.execute("PRAGMA journal_mode = OFF")
        connection.execute("PRAGMA synchronous = OFF")
        connection.executescript(
            """
            CREATE TABLE arrow_duel_metric_definitions (
              id INTEGER PRIMARY KEY,
              key TEXT NOT NULL UNIQUE,
              bit_width INTEGER NOT NULL,
              description TEXT NOT NULL
            );

            CREATE TABLE arrow_duel_route_metric_chunks (
              chunk_id INTEGER PRIMARY KEY,
              first_puzzle_rowid INTEGER NOT NULL UNIQUE,
              puzzle_count INTEGER NOT NULL CHECK (puzzle_count > 0),
              packed_metrics BLOB NOT NULL,
              CHECK (length(packed_metrics) = (puzzle_count + 1) / 2)
            ) WITHOUT ROWID;

            CREATE TABLE compact_metadata (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            ) WITHOUT ROWID;
            """
        )
        connection.executemany(
            """
            INSERT INTO arrow_duel_metric_definitions
              (id, key, bit_width, description)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    1,
                    "top_move_unique_plies",
                    4,
                    "one nibble per puzzle row: 0 unavailable, 1-max exact "
                    "plies, max+1 censored, remaining codes reserved; odd "
                    "row offsets use the low nibble",
                ),
            ],
        )
        connection.execute("ATTACH DATABASE ? AS pack", (str(input_path.resolve()),))
        connection.execute(
            "ATTACH DATABASE ? AS analysis", (str(analysis_path.resolve()),)
        )
        min_rowid, max_rowid, puzzle_count = connection.execute(
            "SELECT MIN(rowid), MAX(rowid), COUNT(*) FROM pack.puzzles"
        ).fetchone()
        if (
            puzzle_count < 1
            or int(min_rowid) != 1
            or int(max_rowid) != int(puzzle_count)
        ):
            raise ValueError(
                "Compact output requires contiguous Core Pack puzzle rowids from 1"
            )
        metric_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM analysis.arrow_duel_route_metrics"
            ).fetchone()[0]
        )
        cursor = connection.execute(
            """
            SELECT
              p.rowid,
              m.top_move_unique_plies
            FROM pack.puzzles p
            LEFT JOIN analysis.arrow_duel_route_metrics m
              ON m.puzzle_id = p.id
            ORDER BY p.rowid
            """
        )
        encoded_metric_count = 0
        chunk_count = 0
        while rows := cursor.fetchmany(COMPACT_CHUNK_SIZE):
            first_rowid = int(rows[0][0])
            expected_first_rowid = chunk_count * COMPACT_CHUNK_SIZE + 1
            if first_rowid != expected_first_rowid:
                raise ValueError("Core Pack puzzle rowids changed during compact export")
            nibbles = [
                encode_compact_metric(
                    int(route_plies) if route_plies is not None else None
                )
                for _, route_plies in rows
            ]
            payload = bytes(
                nibbles[index]
                | (nibbles[index + 1] << 4 if index + 1 < len(nibbles) else 0)
                for index in range(0, len(nibbles), 2)
            )
            encoded_metric_count += sum(value != 0 for value in nibbles)
            connection.execute(
                """
                INSERT INTO arrow_duel_route_metric_chunks (
                  chunk_id,
                  first_puzzle_rowid,
                  puzzle_count,
                  packed_metrics
                ) VALUES (?, ?, ?, ?)
                """,
                (chunk_count, first_rowid, len(rows), payload),
            )
            chunk_count += 1
        if encoded_metric_count != metric_count:
            raise ValueError(
                "Compact output metric count does not match the analysis sidecar"
            )
        metadata = {
            "encoding_version": str(COMPACT_ENCODING_VERSION),
            "input_sha256": str(identity["inputSha256"]),
            "input_puzzle_count": str(puzzle_count),
            "arrow_duel_metric_count": str(metric_count),
            "chunk_size": str(COMPACT_CHUNK_SIZE),
            "max_unique_plies": str(config.max_unique_plies),
            "censored_top_move_unique_code": str(config.max_unique_plies + 1),
            "unique_followup_moves_formula": (
                "floor((top_move_unique_plies - 1) / 2), excluding the "
                "displayed top move; "
                f"{(config.max_unique_plies - 1) // 2}+ is the censored bucket"
            ),
            "analysis_identity": json.dumps(
                identity, sort_keys=True, separators=(",", ":")
            ),
        }
        connection.executemany(
            "INSERT INTO compact_metadata (key, value) VALUES (?, ?)",
            sorted(metadata.items()),
        )
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(
                f"Compact route-metric SQLite integrity check failed: {integrity}"
            )
    except Exception:
        connection.close()
        compact_path.unlink(missing_ok=True)
        raise
    else:
        connection.close()
    return {
        "path": str(compact_path.resolve()),
        "bytes": compact_path.stat().st_size,
        "payloadBytes": (int(puzzle_count) + 1) // 2,
        "chunkCount": chunk_count,
        "chunkSize": COMPACT_CHUNK_SIZE,
        "encodingVersion": COMPACT_ENCODING_VERSION,
        "metricCount": metric_count,
    }


def analyze_pack(
    input_path: Path,
    output_path: Path,
    *,
    report_path: Path,
    compact_output_path: Path | None,
    depth: int,
    stockfish_path: str,
    jobs: int,
    threads_per_job: int,
    hash_mb: int,
    batch_size: int,
    progress_every: int,
    config: RouteConfig,
    overwrite: bool,
    resume: bool,
) -> dict[str, object]:
    input_connection = sqlite_read_only(input_path)
    try:
        pack_layout = validate_pack(input_connection)
        pack_puzzle_count = pack_layout.puzzle_count
        arrow_duel_input_count = int(
            input_connection.execute(
                "SELECT COUNT(*) FROM puzzles p WHERE "
                + arrow_duel_input_sql(pack_layout.row_encoding)
            ).fetchone()[0]
        )
    finally:
        input_connection.close()

    input_digest = sha256(input_path)
    with StockfishEngine(stockfish_path, threads_per_job, hash_mb) as engine:
        engine_name = engine.name
        engine_author = engine.author
    identity = analysis_identity(
        input_path=input_path,
        input_sha256=input_digest,
        depth=depth,
        stockfish_path=stockfish_path,
        engine_name=engine_name,
        engine_author=engine_author,
        threads_per_job=threads_per_job,
        hash_mb=hash_mb,
        jobs=jobs,
        batch_size=batch_size,
        row_encoding=pack_layout.row_encoding,
        config=config,
    )
    output, last_id, processed = initialize_output(
        output_path,
        identity,
        overwrite=overwrite,
        resume=resume,
    )
    started_at = datetime.now(timezone.utc)
    last_progress = processed
    input_connection = sqlite_read_only(input_path)
    window_size = jobs * batch_size
    try:
        with ThreadPoolExecutor(
            max_workers=jobs,
            initializer=_initialize_worker,
            initargs=(
                stockfish_path,
                threads_per_job,
                hash_mb,
                depth,
                config,
            ),
        ) as executor:
            while True:
                rows = input_connection.execute(
                    """
                    SELECT p.id, p.initial_fen, p.rating, p.stockfish_bestmove
                    FROM puzzles p
                    WHERE p.id > ?
                      AND (
                        """ + arrow_duel_input_sql(pack_layout.row_encoding) + """
                      )
                    ORDER BY p.id ASC
                    LIMIT ?
                    """,
                    (last_id, window_size),
                ).fetchall()
                if not rows:
                    break
                puzzles = [
                    decode_puzzle_row(row, pack_layout.row_encoding) for row in rows
                ]
                metrics = [
                    metric
                    for batch in executor.map(
                        _analyze_batch,
                        chunks(puzzles, batch_size),
                    )
                    for metric in batch
                ]
                last_id = puzzles[-1].puzzle_id
                processed += len(puzzles)
                with output:
                    output.executemany(
                        """
                        INSERT INTO arrow_duel_route_metrics (
                          puzzle_id,
                          top_move_unique_plies
                        ) VALUES (?, ?)
                        ON CONFLICT(puzzle_id) DO UPDATE SET
                          top_move_unique_plies = excluded.top_move_unique_plies
                        """,
                        [
                            (
                                metric.puzzle_id,
                                metric.top_move_unique_plies,
                            )
                            for metric in metrics
                        ],
                    )
                    output.execute(
                        """
                        UPDATE analysis_metadata
                        SET value = ?
                        WHERE key = 'last_puzzle_id'
                        """,
                        (last_id,),
                    )
                    output.execute(
                        """
                        UPDATE analysis_metadata
                        SET value = ?
                        WHERE key = 'processed_count'
                        """,
                        (str(processed),),
                    )
                if (
                    progress_every
                    and processed // progress_every > last_progress // progress_every
                ):
                    elapsed = datetime.now(timezone.utc) - started_at
                    print(
                        f"Processed {processed:,}/{arrow_duel_input_count:,} "
                        f"Arrow Duel inputs; elapsed={elapsed}",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_progress = processed
        _close_worker_engines()
    finally:
        _close_worker_engines()
        input_connection.close()

    ending_input_digest = sha256(input_path)
    if ending_input_digest != input_digest:
        output.close()
        raise ValueError(
            "Core Pack changed during route analysis; keep the input artifact "
            "immutable and resume only with the original SHA-256"
        )

    with output:
        output.execute(
            """
            INSERT OR REPLACE INTO analysis_metadata (key, value)
            VALUES ('completed_at', ?)
            """,
            (datetime.now(timezone.utc).isoformat(),),
        )
    output.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    output.execute("VACUUM")
    integrity = output.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise RuntimeError(f"Route-metric SQLite integrity check failed: {integrity}")
    output.close()

    compact_output = (
        build_compact_output(
            input_path=input_path,
            analysis_path=output_path,
            compact_path=compact_output_path,
            identity=identity,
            config=config,
            replace_existing=overwrite or resume,
        )
        if compact_output_path is not None
        else None
    )
    report = build_report(
        input_path=input_path,
        output_path=output_path,
        pack_puzzle_count=pack_puzzle_count,
        identity=identity,
        config=config,
    )
    if compact_output is not None:
        report["compactOutput"] = compact_output
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_report(
    *,
    input_path: Path,
    output_path: Path,
    pack_puzzle_count: int,
    identity: dict[str, object],
    config: RouteConfig,
) -> dict[str, object]:
    connection = sqlite3.connect(output_path)
    try:
        connection.execute("ATTACH DATABASE ? AS pack", (str(input_path.resolve()),))
        arrow_duel_input_count = int(
            connection.execute("SELECT COUNT(*) FROM arrow_duel_route_metrics").fetchone()[0]
        )
        top_move_route_distribution = {
            str(int(plies)): int(total)
            for plies, total in connection.execute(
                """
                SELECT top_move_unique_plies, COUNT(*)
                FROM arrow_duel_route_metrics
                GROUP BY top_move_unique_plies
                ORDER BY top_move_unique_plies
                """
            )
        }
        censored_followups = (config.max_unique_plies - 1) // 2
        followup_distribution: dict[str, int] = {}
        for plies_text, total in top_move_route_distribution.items():
            plies = int(plies_text)
            followups = min((plies - 1) // 2, censored_followups)
            key = (
                f"{censored_followups}+"
                if followups >= censored_followups
                else str(followups)
            )
            followup_distribution[key] = followup_distribution.get(key, 0) + total
        rating_metric_rows = connection.execute(
            """
            SELECT
              CAST(p.rating / 100 AS INTEGER) * 100 AS rating_min,
              m.top_move_unique_plies,
              COUNT(*) AS total
            FROM arrow_duel_route_metrics m
            JOIN pack.puzzles p ON p.id = m.puzzle_id
            GROUP BY rating_min, m.top_move_unique_plies
            ORDER BY rating_min, m.top_move_unique_plies
            """
        ).fetchall()
        rating_distributions: dict[int, dict[str, dict[str, int]]] = {}
        for rating_min, plies, total in rating_metric_rows:
            bucket = rating_distributions.setdefault(
                int(rating_min),
                {
                    "topMoveUniquePliesDistribution": {},
                    "uniqueFollowupMovesDistribution": {},
                },
            )
            plies_key = str(int(plies))
            bucket["topMoveUniquePliesDistribution"][plies_key] = int(total)
            followups = min((int(plies) - 1) // 2, censored_followups)
            followup_key = (
                f"{censored_followups}+"
                if followups >= censored_followups
                else str(followups)
            )
            followup_counts = bucket["uniqueFollowupMovesDistribution"]
            followup_counts[followup_key] = (
                followup_counts.get(followup_key, 0) + int(total)
            )
        rating_buckets = [
            {
                "minRating": int(rating_min),
                "maxRating": int(rating_min) + 99,
                "arrowDuelEligible": sum(
                    rating_distributions[int(rating_min)][
                        "topMoveUniquePliesDistribution"
                    ].values()
                ),
                **rating_distributions[int(rating_min)],
            }
            for rating_min in sorted(rating_distributions)
        ]
        samples: dict[str, list[dict[str, int | str]]] = {}
        for plies in range(1, config.max_unique_plies + 2):
            rows = connection.execute(
                """
                SELECT m.puzzle_id, p.rating
                FROM arrow_duel_route_metrics m
                JOIN pack.puzzles p ON p.id = m.puzzle_id
                WHERE m.top_move_unique_plies = ?
                ORDER BY m.puzzle_id
                LIMIT 20
                """,
                (plies,),
            ).fetchall()
            samples[str(plies)] = [
                {
                    "puzzleId": puzzle_id,
                    "rating": int(rating),
                }
                for puzzle_id, rating in rows
            ]
        followup_samples: dict[str, list[dict[str, int | str]]] = {}
        for followups in range(censored_followups + 1):
            followup_key = (
                f"{censored_followups}+"
                if followups == censored_followups
                else str(followups)
            )
            comparison = (
                "m.top_move_unique_plies >= ?"
                if followups == censored_followups
                else "CAST((m.top_move_unique_plies - 1) / 2 AS INTEGER) = ?"
            )
            comparison_value = (
                2 * censored_followups + 1
                if followups == censored_followups
                else followups
            )
            rows = connection.execute(
                f"""
                WITH eligible AS (
                  SELECT
                    m.puzzle_id,
                    p.rating,
                    m.top_move_unique_plies
                  FROM arrow_duel_route_metrics m
                  JOIN pack.puzzles p ON p.id = m.puzzle_id
                  WHERE {comparison}
                ),
                bucketed AS (
                  SELECT
                    *,
                    NTILE(5) OVER (ORDER BY rating, puzzle_id) AS rating_bucket
                  FROM eligible
                ),
                sampled AS (
                  SELECT
                    *,
                    ROW_NUMBER() OVER (
                      PARTITION BY rating_bucket ORDER BY puzzle_id
                    ) AS sample_rank
                  FROM bucketed
                )
                SELECT puzzle_id, rating, top_move_unique_plies
                FROM sampled
                WHERE sample_rank = 1
                ORDER BY rating_bucket
                """,
                (comparison_value,),
            ).fetchall()
            followup_samples[followup_key] = [
                {
                    "puzzleId": puzzle_id,
                    "rating": int(rating),
                    "topMoveUniquePlies": int(plies),
                }
                for puzzle_id, rating, plies in rows
            ]
        regressions = {}
        for puzzle_id in ("RVpOb", "02ARe", "00SMl", "01N0w"):
            row = connection.execute(
                """
                SELECT top_move_unique_plies
                FROM arrow_duel_route_metrics
                WHERE puzzle_id = ?
                """,
                (puzzle_id,),
            ).fetchone()
            regressions[puzzle_id] = (
                None
                if row is None
                else {
                    "topMoveUniquePlies": int(row[0]),
                    "uniqueFollowupMoves": (
                        f"{censored_followups}+"
                        if int(row[0]) == config.max_unique_plies + 1
                        else (int(row[0]) - 1) // 2
                    ),
                }
            )
    finally:
        connection.close()

    return {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "input": identity,
        "packPuzzleCount": pack_puzzle_count,
        "arrowDuelInputCount": arrow_duel_input_count,
        "corePackRowsExcludedBeforeAnalysis": pack_puzzle_count
        - arrow_duel_input_count,
        "topMoveUniquePliesDistribution": top_move_route_distribution,
        "uniqueFollowupMovesDistribution": followup_distribution,
        "topMoveUniqueCensoredCode": config.max_unique_plies + 1,
        "uniqueFollowupCensoredMinimum": censored_followups,
        "analysisSidecarBytes": output_path.stat().st_size,
        "ratingBuckets": rating_buckets,
        "samples": samples,
        "samplesByUniqueFollowupMoves": followup_samples,
        "regressions": regressions,
    }


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_pack", type=Path)
    parser.add_argument("output_sqlite", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--compact-output",
        type=Path,
        help="optional two-Core-Pack-rows-per-byte SQLite output",
    )
    parser.add_argument("--depth", type=positive_int, default=10)
    parser.add_argument("--stockfish-path", default="stockfish")
    parser.add_argument(
        "--jobs", type=positive_int, default=min(8, max(1, os.cpu_count() or 1))
    )
    parser.add_argument("--threads-per-job", type=positive_int, default=1)
    parser.add_argument("--hash-mb", type=positive_int, default=4)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--progress-every", type=nonnegative_int, default=10_000)
    parser.add_argument("--multipv", type=positive_int, default=ROUTE_MULTIPV)
    parser.add_argument("--eval-tolerance-cp", type=nonnegative_int, default=100)
    parser.add_argument("--max-unique-plies", type=positive_int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive")
    if args.report is None:
        args.report = args.output_sqlite.with_suffix(".report.json")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        config = RouteConfig(
            multipv=args.multipv,
            eval_tolerance_cp=args.eval_tolerance_cp,
            max_unique_plies=args.max_unique_plies,
        )
        report = analyze_pack(
            args.input_pack,
            args.output_sqlite,
            report_path=args.report,
            compact_output_path=args.compact_output,
            depth=args.depth,
            stockfish_path=args.stockfish_path,
            jobs=args.jobs,
            threads_per_job=args.threads_per_job,
            hash_mb=args.hash_mb,
            batch_size=args.batch_size,
            progress_every=args.progress_every,
            config=config,
            overwrite=args.overwrite,
            resume=args.resume,
        )
    except (OSError, ValueError, sqlite3.Error, StockfishError) as error:
        raise SystemExit(f"error: {error}") from error
    print(
        f"Analyzed {report['arrowDuelInputCount']:,} Arrow Duel positions",
        file=sys.stderr,
    )
    print(f"Wrote route metrics to {args.output_sqlite}", file=sys.stderr)
    if args.compact_output is not None:
        print(f"Wrote compact metrics to {args.compact_output}", file=sys.stderr)
    print(f"Wrote report to {args.report}", file=sys.stderr)


if __name__ == "__main__":
    main()
