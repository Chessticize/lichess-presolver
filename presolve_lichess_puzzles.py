#!/usr/bin/env python3
"""Add reproducible Stockfish evaluations to a Lichess puzzle CSV snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, TextIO


SOURCE_COLUMNS = (
    "PuzzleId",
    "FEN",
    "Moves",
    "Rating",
    "RatingDeviation",
    "Popularity",
    "NbPlays",
    "Themes",
    "GameUrl",
    "OpeningTags",
)
OUTPUT_COLUMNS = (
    "stockfish_eval",
    "stockfish_bestmove",
    "stockfish_eval_after_first_move",
)
MATE_SCORE_CP = 10000


class StockfishError(RuntimeError):
    """Raised when the Stockfish process violates the expected UCI contract."""


@dataclass(frozen=True)
class Evaluation:
    score_cp: int
    bestmove: str
    principal_variation: tuple[str, ...] = ()
    depth: int = 0
    mate_in: int | None = None


def score_from_uci_tokens(tokens: list[str]) -> int | None:
    """Return a centipawn-like score from one tokenized UCI info line."""
    if "score" not in tokens:
        return None
    index = tokens.index("score")
    if index + 2 >= len(tokens):
        return None
    score_type = tokens[index + 1]
    try:
        value = int(tokens[index + 2])
    except ValueError:
        return None
    if score_type == "cp":
        return value
    if score_type == "mate":
        return MATE_SCORE_CP if value > 0 else -MATE_SCORE_CP
    return None


def mate_from_uci_tokens(tokens: list[str]) -> int | None:
    """Return Stockfish's signed mate distance from one UCI info line."""
    if "score" not in tokens:
        return None
    index = tokens.index("score")
    if index + 2 >= len(tokens) or tokens[index + 1] != "mate":
        return None
    try:
        return int(tokens[index + 2])
    except ValueError:
        return None


def score_to_white_perspective(score: int, fen: str, applied_moves: int = 0) -> int:
    """Convert a side-to-move UCI score to White's point of view."""
    fen_parts = fen.split()
    if len(fen_parts) < 2 or fen_parts[1] not in {"w", "b"}:
        raise ValueError(f"Invalid FEN active-color field: {fen!r}")
    white_to_move = fen_parts[1] == "w"
    if applied_moves % 2:
        white_to_move = not white_to_move
    return score if white_to_move else -score


class StockfishEngine:
    """A persistent, single-process UCI Stockfish client."""

    def __init__(self, executable: str, threads: int, hash_mb: int) -> None:
        try:
            self.process = subprocess.Popen(
                [executable],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError as error:
            raise StockfishError(f"Could not start Stockfish at {executable!r}: {error}") from error

        self.name = "unknown"
        self.author = "unknown"
        self._send("uci")
        self._initialize()
        self._send(f"setoption name Threads value {threads}")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send("setoption name UCI_Chess960 value false")
        self._send("isready")
        self._wait_for("readyok")
        self.start_new_game()

    @property
    def stdin(self) -> TextIO:
        if self.process.stdin is None:
            raise StockfishError("Stockfish stdin is unavailable")
        return self.process.stdin

    @property
    def stdout(self) -> TextIO:
        if self.process.stdout is None:
            raise StockfishError("Stockfish stdout is unavailable")
        return self.process.stdout

    def _send(self, command: str) -> None:
        if self.process.poll() is not None:
            raise StockfishError("Stockfish exited unexpectedly")
        self.stdin.write(command + "\n")
        self.stdin.flush()

    def _initialize(self) -> None:
        for raw_line in self.stdout:
            line = raw_line.strip()
            if line.startswith("id name "):
                self.name = line.removeprefix("id name ")
            elif line.startswith("id author "):
                self.author = line.removeprefix("id author ")
            elif line == "uciok":
                return
        raise StockfishError("Stockfish exited before completing UCI initialization")

    def _wait_for(self, expected: str) -> None:
        for line in self.stdout:
            if line.strip() == expected:
                return
        raise StockfishError(f"Stockfish exited before returning {expected!r}")

    def start_new_game(self) -> None:
        """Reset engine search state before an independent position."""
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok")

    def evaluate(
        self,
        fen: str,
        depth: int,
        moves: Iterable[str] = (),
        searchmoves: Iterable[str] = (),
    ) -> Evaluation:
        move_list = list(moves)
        search_move_list = list(searchmoves)
        suffix = f" moves {' '.join(move_list)}" if move_list else ""
        self._send(f"position fen {fen}{suffix}")
        search_suffix = (
            f" searchmoves {' '.join(search_move_list)}" if search_move_list else ""
        )
        self._send(f"go depth {depth}{search_suffix}")
        score: int | None = None
        mate_in: int | None = None
        principal_variation: tuple[str, ...] = ()
        principal_variation_depth = 0

        for line in self.stdout:
            tokens = line.split()
            if tokens[:1] == ["info"]:
                candidate = score_from_uci_tokens(tokens)
                if candidate is not None:
                    score = candidate
                    raw_mate = mate_from_uci_tokens(tokens)
                    mate_in = (
                        score_to_white_perspective(raw_mate, fen, len(move_list))
                        if raw_mate is not None
                        else None
                    )
                if "pv" in tokens:
                    pv_index = tokens.index("pv")
                    depth_index = tokens.index("depth") if "depth" in tokens else -1
                    try:
                        info_depth = int(tokens[depth_index + 1]) if depth_index >= 0 else 0
                    except (IndexError, ValueError):
                        info_depth = 0
                    if info_depth >= principal_variation_depth:
                        principal_variation = tuple(tokens[pv_index + 1 :])
                        principal_variation_depth = info_depth
            elif tokens[:1] == ["bestmove"]:
                if score is None or len(tokens) < 2 or tokens[1] == "(none)":
                    raise StockfishError(f"Incomplete Stockfish result for FEN {fen!r}")
                return Evaluation(
                    score_cp=score_to_white_perspective(score, fen, len(move_list)),
                    bestmove=tokens[1],
                    principal_variation=principal_variation,
                    depth=principal_variation_depth,
                    mate_in=mate_in,
                )

        raise StockfishError("Stockfish exited before returning a best move")

    def evaluate_multipv(
        self,
        fen: str,
        depth: int,
        *,
        multipv: int,
        moves: Iterable[str] = (),
    ) -> tuple[Evaluation, ...]:
        """Return distinct root continuations ordered by Stockfish preference."""
        if multipv < 1:
            raise ValueError("multipv must be at least 1")
        move_list = list(moves)
        suffix = f" moves {' '.join(move_list)}" if move_list else ""
        self._send(f"setoption name MultiPV value {multipv}")
        try:
            self._send(f"position fen {fen}{suffix}")
            self._send(f"go depth {depth}")
            lines: dict[int, Evaluation] = {}
            for line in self.stdout:
                tokens = line.split()
                if tokens[:1] == ["info"] and "pv" in tokens:
                    score = score_from_uci_tokens(tokens)
                    if score is None:
                        continue
                    pv_index = tokens.index("pv")
                    principal_variation = tuple(tokens[pv_index + 1 :])
                    if not principal_variation:
                        continue
                    multipv_index = tokens.index("multipv") if "multipv" in tokens else -1
                    depth_index = tokens.index("depth") if "depth" in tokens else -1
                    try:
                        line_number = (
                            int(tokens[multipv_index + 1]) if multipv_index >= 0 else 1
                        )
                        info_depth = (
                            int(tokens[depth_index + 1]) if depth_index >= 0 else 0
                        )
                    except (IndexError, ValueError):
                        continue
                    existing = lines.get(line_number)
                    if existing is not None and existing.depth > info_depth:
                        continue
                    lines[line_number] = Evaluation(
                        score_cp=score_to_white_perspective(
                            score, fen, len(move_list)
                        ),
                        bestmove=principal_variation[0],
                        principal_variation=principal_variation,
                        depth=info_depth,
                        mate_in=(
                            score_to_white_perspective(
                                raw_mate, fen, len(move_list)
                            )
                            if (raw_mate := mate_from_uci_tokens(tokens)) is not None
                            else None
                        ),
                    )
                elif tokens[:1] == ["bestmove"]:
                    break
            if not lines:
                raise StockfishError(
                    f"Stockfish returned no MultiPV lines for FEN {fen!r}"
                )
            return tuple(lines[index] for index in sorted(lines))
        finally:
            self._send("setoption name MultiPV value 1")

    def is_checkmate(self, fen: str, moves: Iterable[str] = ()) -> bool:
        """Return whether the reached position is terminal checkmate."""
        move_list = list(moves)
        suffix = f" moves {' '.join(move_list)}" if move_list else ""
        self._send(f"position fen {fen}{suffix}")
        self._send("go depth 1")
        saw_mate_zero = False
        for line in self.stdout:
            tokens = line.split()
            if tokens[:1] == ["info"]:
                saw_mate_zero = saw_mate_zero or mate_from_uci_tokens(tokens) == 0
            elif tokens[:1] == ["bestmove"]:
                no_move = len(tokens) < 2 or tokens[1] in {"(none)", "0000"}
                return no_move and saw_mate_zero
        raise StockfishError("Stockfish exited before returning a best move")

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._send("quit")
                self.process.wait(timeout=5)
            except (BrokenPipeError, StockfishError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None and not stream.closed:
                stream.close()

    def __enter__(self) -> StockfishEngine:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_puzzles(
    input_csv: Path,
    output_csv: Path,
    *,
    depth: int,
    stockfish_path: str,
    threads: int,
    hash_mb: int,
    progress_every: int,
    overwrite: bool,
    write_metadata: bool,
) -> int:
    if input_csv.resolve() == output_csv.resolve():
        raise ValueError("Input and output CSV paths must differ")
    if output_csv.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_csv}; pass --overwrite to replace it")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    partial_csv = output_csv.with_name(output_csv.name + ".partial")
    if partial_csv.exists() and not overwrite:
        raise FileExistsError(
            f"Partial output already exists: {partial_csv}; pass --overwrite to replace it"
        )

    started_at = datetime.now(timezone.utc)
    row_count = 0
    engine_name = "unknown"
    engine_author = "unknown"

    try:
        with (
            input_csv.open(newline="", encoding="utf-8-sig") as input_file,
            partial_csv.open("w", newline="", encoding="utf-8") as output_file,
            StockfishEngine(stockfish_path, threads, hash_mb) as engine,
        ):
            reader = csv.DictReader(input_file)
            fieldnames = reader.fieldnames or []
            missing = [column for column in SOURCE_COLUMNS if column not in fieldnames]
            existing_outputs = [column for column in OUTPUT_COLUMNS if column in fieldnames]
            if missing:
                raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")
            if existing_outputs:
                raise ValueError(
                    "Input CSV already contains generated columns: "
                    + ", ".join(existing_outputs)
                )

            engine_name = engine.name
            engine_author = engine.author
            writer = csv.DictWriter(output_file, fieldnames=[*fieldnames, *OUTPUT_COLUMNS])
            writer.writeheader()

            for row_count, row in enumerate(reader, 1):
                moves = row["Moves"].split()
                if not moves:
                    raise ValueError(f"Puzzle {row.get('PuzzleId', row_count)!r} has no moves")
                initial = engine.evaluate(row["FEN"], depth)
                after_first_move = engine.evaluate(row["FEN"], depth, [moves[0]])
                row["stockfish_eval"] = initial.score_cp
                row["stockfish_bestmove"] = initial.bestmove
                row["stockfish_eval_after_first_move"] = after_first_move.score_cp
                writer.writerow(row)

                if progress_every and row_count % progress_every == 0:
                    elapsed = datetime.now(timezone.utc) - started_at
                    print(
                        f"Processed {row_count:,} puzzles in {elapsed}",
                        file=sys.stderr,
                        flush=True,
                    )
        os.replace(partial_csv, output_csv)
    except BaseException:
        if partial_csv.exists():
            print(f"Partial output retained at {partial_csv}", file=sys.stderr)
        raise

    if write_metadata:
        metadata = {
            "schema_version": 1,
            "source": {
                "path": str(input_csv),
                "sha256": sha256(input_csv),
            },
            "output": {
                "path": str(output_csv),
                "sha256": sha256(output_csv),
                "rows": row_count,
            },
            "stockfish": {
                "name": engine_name,
                "author": engine_author,
                "depth": depth,
                "threads": threads,
                "hash_mb": hash_mb,
            },
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        metadata_path = output_csv.with_name(output_csv.name + ".metadata.json")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return row_count


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="Lichess puzzle CSV")
    parser.add_argument("output_csv", type=Path, help="Destination CSV")
    parser.add_argument("--depth", type=positive_int, default=20)
    parser.add_argument("--stockfish-path", default="stockfish")
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--hash-mb", type=positive_int, default=128)
    parser.add_argument("--progress-every", type=nonnegative_int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-metadata", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        rows = process_puzzles(
            args.input_csv,
            args.output_csv,
            depth=args.depth,
            stockfish_path=args.stockfish_path,
            threads=args.threads,
            hash_mb=args.hash_mb,
            progress_every=args.progress_every,
            overwrite=args.overwrite,
            write_metadata=not args.no_metadata,
        )
    except (OSError, ValueError, StockfishError) as error:
        raise SystemExit(f"error: {error}") from error
    print(f"Wrote {rows:,} puzzles to {args.output_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
