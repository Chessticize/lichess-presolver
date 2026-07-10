#!/usr/bin/env python3
"""Split a CSV into round-robin parts while preserving its header."""

from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
from pathlib import Path


def split_csv(input_csv: Path, num_splits: int, output_prefix: Path) -> list[Path]:
    if num_splits < 1:
        raise ValueError("num_splits must be at least 1")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_paths = [
        output_prefix.with_name(f"{output_prefix.name}{index}.csv")
        for index in range(1, num_splits + 1)
    ]
    if input_csv.resolve() in {path.resolve() for path in output_paths}:
        raise ValueError("An output path would overwrite the input CSV")

    with ExitStack() as stack:
        input_file = stack.enter_context(input_csv.open(newline="", encoding="utf-8-sig"))
        reader = csv.reader(input_file)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("Input CSV is empty") from error

        writers = []
        for output_path in output_paths:
            output_file = stack.enter_context(
                output_path.open("w", newline="", encoding="utf-8")
            )
            writer = csv.writer(output_file)
            writer.writerow(header)
            writers.append(writer)

        for row_index, row in enumerate(reader):
            writers[row_index % num_splits].writerow(row)

    return output_paths


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("num_splits", type=positive_int)
    parser.add_argument(
        "--output-prefix",
        "--output_prefix",
        type=Path,
        default=Path("split_"),
        help="Output prefix; both spellings are accepted for backward compatibility",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        paths = split_csv(args.input_csv, args.num_splits, args.output_prefix)
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(f"Wrote {len(paths)} files with prefix {args.output_prefix}")


if __name__ == "__main__":
    main()
