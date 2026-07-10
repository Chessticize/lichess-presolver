#!/usr/bin/env python3
"""Presolve a directory of puzzle CSV splits with parallel Stockfish workers."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("puzzles"))
    parser.add_argument("--output-dir", type=Path, default=Path("presolved-depth20"))
    parser.add_argument("--pattern", default="split_*.csv")
    parser.add_argument("--depth", type=positive_int, default=20)
    parser.add_argument(
        "--jobs", type=positive_int, default=min(8, max(1, os.cpu_count() or 1))
    )
    parser.add_argument("--threads-per-job", type=positive_int, default=1)
    parser.add_argument("--hash-mb", type=positive_int, default=128)
    parser.add_argument("--stockfish-path", default="stockfish")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def command_for(input_path: Path, output_path: Path, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("presolve_lichess_puzzles.py")),
        str(input_path),
        str(output_path),
        "--depth",
        str(args.depth),
        "--stockfish-path",
        args.stockfish_path,
        "--threads",
        str(args.threads_per_job),
        "--hash-mb",
        str(args.hash_mb),
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def main() -> None:
    args = parse_args()
    inputs = sorted(args.input_dir.glob(args.pattern))
    if not inputs:
        raise SystemExit(
            f"error: no files matching {args.pattern!r} in {args.input_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Processing {len(inputs)} files at depth {args.depth} with {args.jobs} workers",
        file=sys.stderr,
    )
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                subprocess.run,
                command_for(input_path, args.output_dir / input_path.name, args),
                check=False,
            ): input_path
            for input_path in inputs
        }
        for future in as_completed(futures):
            input_path = futures[future]
            return_code = future.result().returncode
            if return_code:
                failures.append(f"{input_path} (exit {return_code})")
            else:
                print(f"Completed {input_path}", file=sys.stderr)

    if failures:
        raise SystemExit("error: failed inputs: " + ", ".join(failures))


if __name__ == "__main__":
    main()
