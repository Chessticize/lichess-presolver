from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from presolve_lichess_puzzles import (  # noqa: E402
    MATE_SCORE_CP,
    score_from_uci_tokens,
    score_to_white_perspective,
)
from tools.compare_outputs import compare  # noqa: E402


PUZZLE_ROW = {
    "PuzzleId": "00sHx",
    "FEN": "q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17",
    "Moves": "e8d7 a2e6 d7d8 f7f8",
    "Rating": "1760",
    "RatingDeviation": "80",
    "Popularity": "83",
    "NbPlays": "72",
    "Themes": "mate mateIn2 middlegame short",
    "GameUrl": "https://lichess.org/yyznGmXs/black#34",
    "OpeningTags": "Italian_Game Italian_Game_Classical_Variation",
}


class ScoreParsingTests(unittest.TestCase):
    def test_parses_centipawn_score(self) -> None:
        self.assertEqual(
            score_from_uci_tokens("info depth 8 score cp -123 nodes 1".split()), -123
        )

    def test_converts_mate_score(self) -> None:
        self.assertEqual(
            score_from_uci_tokens("info depth 8 score mate 2 nodes 1".split()),
            MATE_SCORE_CP,
        )
        self.assertEqual(
            score_from_uci_tokens("info depth 8 score mate -2 nodes 1".split()),
            -MATE_SCORE_CP,
        )

    def test_normalizes_to_white_perspective(self) -> None:
        white_fen = "8/8/8/8/8/8/8/K6k w - - 0 1"
        black_fen = "8/8/8/8/8/8/8/K6k b - - 0 1"
        self.assertEqual(score_to_white_perspective(50, white_fen), 50)
        self.assertEqual(score_to_white_perspective(50, black_fen), -50)
        self.assertEqual(score_to_white_perspective(50, black_fen, 1), 50)


class PublicCliTests(unittest.TestCase):
    def test_split_cli_preserves_all_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.csv"
            with source.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=list(PUZZLE_ROW))
                writer.writeheader()
                for index in range(5):
                    writer.writerow({**PUZZLE_ROW, "PuzzleId": f"id-{index}"})

            subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "split_csv_util.py"),
                    str(source),
                    "3",
                    "--output-prefix",
                    str(root / "split_"),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            ids = []
            for path in sorted(root.glob("split_*.csv")):
                with path.open(newline="", encoding="utf-8") as input_file:
                    ids.extend(row["PuzzleId"] for row in csv.DictReader(input_file))
            self.assertCountEqual(ids, [f"id-{index}" for index in range(5)])

    @unittest.skipUnless(shutil.which("stockfish"), "Stockfish is not installed")
    def test_presolver_cli_uses_real_stockfish_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.csv"
            output = root / "output.csv"
            with source.open("w", newline="", encoding="utf-8") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=list(PUZZLE_ROW))
                writer.writeheader()
                writer.writerow(PUZZLE_ROW)

            subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "presolve_lichess_puzzles.py"),
                    str(source),
                    str(output),
                    "--depth",
                    "1",
                    "--hash-mb",
                    "16",
                    "--progress-every",
                    "0",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            with output.open(newline="", encoding="utf-8") as input_file:
                rows = list(csv.DictReader(input_file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["PuzzleId"], PUZZLE_ROW["PuzzleId"])
            self.assertTrue(rows[0]["stockfish_bestmove"])
            int(rows[0]["stockfish_eval"])
            int(rows[0]["stockfish_eval_after_first_move"])

            metadata = json.loads(
                output.with_name("output.csv.metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["output"]["rows"], 1)
            self.assertEqual(metadata["stockfish"]["depth"], 1)
            self.assertIn("Stockfish", metadata["stockfish"]["name"])

            second_run = subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "presolve_lichess_puzzles.py"),
                    str(source),
                    str(output),
                    "--depth",
                    "1",
                    "--progress-every",
                    "0",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(second_run.returncode, 0)
            self.assertIn("pass --overwrite", second_run.stderr)

    @unittest.skipUnless(shutil.which("stockfish"), "Stockfish is not installed")
    def test_parallel_cli_runs_real_worker_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            with (input_dir / "split_1.csv").open(
                "w", newline="", encoding="utf-8"
            ) as output_file:
                writer = csv.DictWriter(output_file, fieldnames=list(PUZZLE_ROW))
                writer.writeheader()
                writer.writerow(PUZZLE_ROW)

            subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "solve_parallel.py"),
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                    "--jobs",
                    "1",
                    "--depth",
                    "1",
                    "--hash-mb",
                    "16",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertTrue((output_dir / "split_1.csv").is_file())
            self.assertTrue((output_dir / "split_1.csv.metadata.json").is_file())


class DatasetComparisonTests(unittest.TestCase):
    def test_compare_streams_matching_source_rows(self) -> None:
        fieldnames = [
            *PUZZLE_ROW,
            "stockfish_eval",
            "stockfish_bestmove",
            "stockfish_eval_after_first_move",
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            left_dir = root / "left"
            right_dir = root / "right"
            left_dir.mkdir()
            right_dir.mkdir()

            for directory, evaluation, bestmove in (
                (left_dir, "10", "e8d7"),
                (right_dir, "30", "e8f7"),
            ):
                with (directory / "split_1.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as output:
                    writer = csv.DictWriter(output, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(
                        {
                            **PUZZLE_ROW,
                            "stockfish_eval": evaluation,
                            "stockfish_bestmove": bestmove,
                            "stockfish_eval_after_first_move": "10000",
                        }
                    )

            report, samples = compare(left_dir, right_dir, sample_size=1, seed=1)
            self.assertEqual(report["rows"], 1)
            self.assertEqual(report["mismatched_source_rows"], 0)
            self.assertEqual(report["bestmove"]["disagreements"], 1)
            self.assertEqual(
                report["evaluation_deltas"]["stockfish_eval"][
                    "mean_absolute_delta_cp"
                ],
                20.0,
            )
            self.assertEqual(len(samples), 1)


if __name__ == "__main__":
    unittest.main()
