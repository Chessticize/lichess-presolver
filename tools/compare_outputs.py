#!/usr/bin/env python3
"""Compare two directories of presolved Lichess puzzle CSV files.

The comparison is streaming, so it can audit the complete multi-gigabyte
snapshot without loading it into memory. An optional deterministic sample of
disagreements can be checked against a local Stockfish executable.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Iterable, TextIO


OUTPUT_COLUMNS = (
    "stockfish_eval",
    "stockfish_bestmove",
    "stockfish_eval_after_first_move",
)


def numeric(value: str) -> int:
    return int(float(value))


def percentile(histogram: Counter[int], percentile_value: float) -> int | None:
    if not histogram:
        return None
    target = max(1, round(sum(histogram.values()) * percentile_value))
    seen = 0
    for value in sorted(histogram):
        seen += histogram[value]
        if seen >= target:
            return value
    raise AssertionError("unreachable")


@dataclass
class DeltaStats:
    count: int = 0
    total: int = 0
    exact: int = 0
    histogram: Counter[int] | None = None

    def __post_init__(self) -> None:
        self.histogram = Counter()

    def add(self, left: str, right: str) -> None:
        delta = abs(numeric(left) - numeric(right))
        self.count += 1
        self.total += delta
        self.exact += delta == 0
        assert self.histogram is not None
        self.histogram[delta] += 1

    def report(self) -> dict[str, float | int | None]:
        assert self.histogram is not None
        return {
            "comparable_rows": self.count,
            "exact_rows": self.exact,
            "mean_absolute_delta_cp": round(self.total / self.count, 3)
            if self.count
            else None,
            "median_absolute_delta_cp": percentile(self.histogram, 0.5),
            "p95_absolute_delta_cp": percentile(self.histogram, 0.95),
        }


class UciEngine:
    def __init__(self, executable: str, hash_mb: int = 128) -> None:
        self.process = subprocess.Popen(
            [executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")
        self._send("setoption name Threads value 1")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send("isready")
        self._wait_for("readyok")

    @property
    def stdin(self) -> TextIO:
        if self.process.stdin is None:
            raise RuntimeError("Stockfish stdin is unavailable")
        return self.process.stdin

    @property
    def stdout(self) -> TextIO:
        if self.process.stdout is None:
            raise RuntimeError("Stockfish stdout is unavailable")
        return self.process.stdout

    def _send(self, command: str) -> None:
        self.stdin.write(command + "\n")
        self.stdin.flush()

    def _wait_for(self, expected: str) -> None:
        for line in self.stdout:
            if line.strip() == expected:
                return
        raise RuntimeError(f"Stockfish exited before {expected!r}")

    def evaluate(
        self, fen: str, depth: int, moves: Iterable[str] = ()
    ) -> tuple[int, str]:
        move_list = list(moves)
        suffix = f" moves {' '.join(move_list)}" if move_list else ""
        self._send(f"position fen {fen}{suffix}")
        self._send(f"go depth {depth}")
        score: int | None = None
        bestmove: str | None = None

        for line in self.stdout:
            tokens = line.split()
            if tokens[:1] == ["info"] and "score" in tokens:
                score_index = tokens.index("score")
                if score_index + 2 < len(tokens):
                    score_type = tokens[score_index + 1]
                    score_value = int(tokens[score_index + 2])
                    if score_type == "cp":
                        score = score_value
                    elif score_type == "mate":
                        score = 10000 if score_value > 0 else -10000
            elif tokens[:1] == ["bestmove"]:
                bestmove = tokens[1]
                break

        if score is None or bestmove is None:
            raise RuntimeError("Stockfish returned no final score or best move")

        side_to_move_is_white = fen.split()[1] == "w"
        if len(move_list) % 2:
            side_to_move_is_white = not side_to_move_is_white
        return (score if side_to_move_is_white else -score), bestmove

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        self._send("quit")
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def __enter__(self) -> UciEngine:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def csv_paths(directory: Path) -> dict[str, Path]:
    return {path.name: path for path in directory.glob("*.csv")}


def reservoir_add(
    reservoir: list[tuple[dict[str, str], dict[str, str]]],
    value: tuple[dict[str, str], dict[str, str]],
    seen: int,
    capacity: int,
    rng: random.Random,
) -> None:
    if capacity == 0:
        return
    if len(reservoir) < capacity:
        reservoir.append(value)
        return
    replacement = rng.randrange(seen)
    if replacement < capacity:
        reservoir[replacement] = value


def compare(
    left_dir: Path,
    right_dir: Path,
    sample_size: int,
    seed: int,
) -> tuple[dict[str, object], list[tuple[dict[str, str], dict[str, str]]]]:
    left_paths = csv_paths(left_dir)
    right_paths = csv_paths(right_dir)
    names = sorted(left_paths.keys() | right_paths.keys())
    rng = random.Random(seed)
    samples: list[tuple[dict[str, str], dict[str, str]]] = []

    rows = 0
    mismatched_source_rows = 0
    missing_rows = 0
    invalid_output_rows = Counter({"left": 0, "right": 0})
    bestmove_matches = 0
    bestmove_disagreements = 0
    mate_status_disagreements = Counter({
        "stockfish_eval": 0,
        "stockfish_eval_after_first_move": 0,
    })
    deltas = {
        "stockfish_eval": DeltaStats(),
        "stockfish_eval_after_first_move": DeltaStats(),
    }

    for name in names:
        if name not in left_paths or name not in right_paths:
            missing_rows += 1
            continue
        with ExitStack() as stack:
            left_file = stack.enter_context(left_paths[name].open(newline="", encoding="utf-8-sig"))
            right_file = stack.enter_context(right_paths[name].open(newline="", encoding="utf-8-sig"))
            left_reader = csv.DictReader(left_file)
            right_reader = csv.DictReader(right_file)
            if left_reader.fieldnames != right_reader.fieldnames:
                raise ValueError(f"schema mismatch in {name}")
            source_columns = [
                column for column in (left_reader.fieldnames or []) if column not in OUTPUT_COLUMNS
            ]

            for left, right in zip_longest(left_reader, right_reader):
                if left is None or right is None:
                    missing_rows += 1
                    continue
                rows += 1
                if any(left[column] != right[column] for column in source_columns):
                    mismatched_source_rows += 1
                    continue

                left_valid = all(left.get(column, "") for column in OUTPUT_COLUMNS)
                right_valid = all(right.get(column, "") for column in OUTPUT_COLUMNS)
                invalid_output_rows["left"] += not left_valid
                invalid_output_rows["right"] += not right_valid
                if not (left_valid and right_valid):
                    continue

                if left["stockfish_bestmove"] == right["stockfish_bestmove"]:
                    bestmove_matches += 1
                else:
                    bestmove_disagreements += 1

                for column in deltas:
                    deltas[column].add(left[column], right[column])
                    left_is_mate = abs(numeric(left[column])) == 10000
                    right_is_mate = abs(numeric(right[column])) == 10000
                    mate_status_disagreements[column] += left_is_mate != right_is_mate

                if left["stockfish_bestmove"] != right["stockfish_bestmove"]:
                    reservoir_add(
                        samples,
                        (dict(left), dict(right)),
                        bestmove_disagreements,
                        sample_size,
                        rng,
                    )

    report: dict[str, object] = {
        "left": str(left_dir),
        "right": str(right_dir),
        "files": names,
        "rows": rows,
        "missing_files_or_rows": missing_rows,
        "mismatched_source_rows": mismatched_source_rows,
        "invalid_output_rows": dict(invalid_output_rows),
        "bestmove": {
            "matches": bestmove_matches,
            "disagreements": bestmove_disagreements,
            "agreement_percent": round(
                100 * bestmove_matches / (bestmove_matches + bestmove_disagreements), 4
            )
            if bestmove_matches + bestmove_disagreements
            else None,
        },
        "evaluation_deltas": {
            column: stats.report() for column, stats in deltas.items()
        },
        "mate_status_disagreements": dict(mate_status_disagreements),
    }
    return report, samples


def mean(values: list[int]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def benchmark(
    samples: list[tuple[dict[str, str], dict[str, str]]],
    executable: str,
    depth: int,
) -> dict[str, object]:
    results = {
        "left_bestmove_matches": 0,
        "right_bestmove_matches": 0,
        "both_bestmove_match": 0,
        "neither_bestmove_matches": 0,
    }
    errors = {
        "left_initial": [],
        "right_initial": [],
        "left_after_first_move": [],
        "right_after_first_move": [],
    }

    with UciEngine(executable) as engine:
        for index, (left, right) in enumerate(samples, 1):
            first_move = left["Moves"].split()[0]
            initial_eval, reference_bestmove = engine.evaluate(left["FEN"], depth)
            after_eval, _ = engine.evaluate(left["FEN"], depth, [first_move])

            left_matches = left["stockfish_bestmove"] == reference_bestmove
            right_matches = right["stockfish_bestmove"] == reference_bestmove
            results["left_bestmove_matches"] += left_matches
            results["right_bestmove_matches"] += right_matches
            results["both_bestmove_match"] += left_matches and right_matches
            results["neither_bestmove_matches"] += not left_matches and not right_matches

            for label, value, reference in (
                ("left_initial", left["stockfish_eval"], initial_eval),
                ("right_initial", right["stockfish_eval"], initial_eval),
                (
                    "left_after_first_move",
                    left["stockfish_eval_after_first_move"],
                    after_eval,
                ),
                (
                    "right_after_first_move",
                    right["stockfish_eval_after_first_move"],
                    after_eval,
                ),
            ):
                candidate = numeric(value)
                if abs(candidate) != 10000 and abs(reference) != 10000:
                    errors[label].append(abs(candidate - reference))

            if index % 10 == 0:
                print(
                    f"Benchmarked {index}/{len(samples)} sampled disagreements",
                    file=sys.stderr,
                    flush=True,
                )

    return {
        "stockfish_executable": executable,
        "reference_depth": depth,
        "sample_size": len(samples),
        **results,
        "mean_absolute_error_cp": {label: mean(values) for label, values in errors.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20250725)
    parser.add_argument("--stockfish", default="stockfish")
    parser.add_argument("--reference-depth", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, samples = compare(args.left, args.right, args.sample_size, args.seed)
    if samples:
        report["reference_benchmark"] = benchmark(
            samples, args.stockfish, args.reference_depth
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
