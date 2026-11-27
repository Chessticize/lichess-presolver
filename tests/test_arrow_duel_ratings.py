from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from rate_arrow_duel_puzzles import (
    DEFAULT_ELO_GRID,
    ArrowDuelPuzzleRow,
    encode_binary_v1_position_to_maia3,
    encode_fen_to_maia3,
    run_rating_job,
    solve_prompted_arrow_duel_batch,
    solve_prompted_arrow_duel_rating,
)


class StubScorer:
    def __init__(self, q_map: dict[str, list[float]]) -> None:
        self.q_map = q_map
        self.model_identity = {"family": "stub", "version": 1}

    def score_batch(self, rows, row_encoding: str, elos):
        return [list(self.q_map[row.puzzle_id]) for row in rows]


class ArrowDuelRatingFormulaTests(unittest.TestCase):
    def test_neutral_intuition_matches_original_lichess_rating(self) -> None:
        # When q(E) == 0.5 and check_rate == 1.0, P_AD(R) = 0.5 + 0.5*0.5 = 0.75 exactly.
        for orig in (800, 1200, 1500, 1850, 2100):
            est = solve_prompted_arrow_duel_rating(orig, DEFAULT_ELO_GRID, [0.5] * 5)
            self.assertEqual(est, orig)

    def test_obvious_good_move_drops_to_floor(self) -> None:
        est = solve_prompted_arrow_duel_rating(2150, DEFAULT_ELO_GRID, [0.90, 0.90, 0.90, 0.90, 0.90])
        self.assertEqual(est, 600)

    def test_deceptive_blunder_raises_rating_above_original(self) -> None:
        est = solve_prompted_arrow_duel_rating(1500, DEFAULT_ELO_GRID, [0.0, 0.0, 0.0, 0.0, 0.0])
        # sigmoid((E - 1500)/173.7178) = 0.75 => E = 1500 + 173.7178 * ln(3) ~= 1691
        self.assertAlmostEqual(est, 1691, delta=2)

    def test_batch_solver_matches_scalar_solver(self) -> None:
        origs = [700, 1100, 1500, 1850, 2150]
        qs = [
            [0.10, 0.25, 0.45, 0.65, 0.85],
            [0.80, 0.85, 0.90, 0.92, 0.95],
            [0.50, 0.50, 0.50, 0.50, 0.50],
            [0.02, 0.05, 0.12, 0.25, 0.40],
            [0.72, 0.76, 0.80, 0.84, 0.88],
        ]
        batch_full = solve_prompted_arrow_duel_batch(origs, DEFAULT_ELO_GRID, qs, check_rate=1.0)
        batch_sprint = solve_prompted_arrow_duel_batch(origs, DEFAULT_ELO_GRID, qs, check_rate=0.75)
        for i in range(len(origs)):
            self.assertEqual(
                int(batch_full[i]),
                solve_prompted_arrow_duel_rating(int(origs[i]), DEFAULT_ELO_GRID, qs[i], check_rate=1.0),
            )
            self.assertEqual(
                int(batch_sprint[i]),
                solve_prompted_arrow_duel_rating(int(origs[i]), DEFAULT_ELO_GRID, qs[i], check_rate=0.75),
            )

    def test_fen_encoder_handles_white_and_black_mirroring(self) -> None:
        # White king on e1 (sq 4), black king on e8 (sq 60)
        arr_w = [[0.0] * 12 for _ in range(64)]
        is_black_w = encode_fen_to_maia3("4k3/8/8/8/8/8/8/4K3 w - - 0 1", arr_w)
        self.assertFalse(is_black_w)
        self.assertEqual(arr_w[4][5], 1.0)   # White King -> own king channel 5 on e1
        self.assertEqual(arr_w[60][11], 1.0)  # Black King -> opp king channel 11 on e8

        # Same position with Black to move: board is rank-mirrored and colors swapped!
        arr_b = [[0.0] * 12 for _ in range(64)]
        is_black_b = encode_fen_to_maia3("4k3/8/8/8/8/8/8/4K3 b - - 0 1", arr_b)
        self.assertTrue(is_black_b)
        self.assertEqual(arr_b[4][5], 1.0)   # Black King mirrored e8->e1, own king channel 5
        self.assertEqual(arr_b[60][11], 1.0)  # White King mirrored e1->e8, opp king channel 11


class ArrowDuelRatingPipelineTests(unittest.TestCase):
    def _create_legacy_pack(self, path: Path) -> None:
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            CREATE TABLE puzzles (
              id TEXT PRIMARY KEY,
              initial_fen TEXT NOT NULL,
              solution_moves TEXT NOT NULL,
              rating INTEGER NOT NULL,
              rating_deviation INTEGER NOT NULL,
              stockfish_eval REAL NOT NULL,
              stockfish_bestmove TEXT NOT NULL,
              stockfish_eval_after_first_move REAL NOT NULL
            );
            """
        )
        rows = [
            ("p1", "4k3/8/8/8/8/8/4R3/4K3 w - -", "e2e1 e8d7", 1500, 75, 500.0, "e2e7", -500.0),
            ("p2", "4k3/8/8/8/8/8/4R3/4K3 w - -", "e2e1 e8d7", 2100, 75, 500.0, "e2e7", -500.0),
            # Promotion candidate should be excluded from Arrow Duel set
            ("p3_promo", "4k3/4P3/8/8/8/8/8/4K3 w - -", "e7e8q e8d7", 1200, 75, 500.0, "e7e8n", -500.0),
        ]
        conn.executemany("INSERT INTO puzzles VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        conn.commit()
        conn.close()

    def test_end_to_end_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = root / "pack.sqlite"
            out = root / "ratings.sqlite"
            report_path = root / "report.json"
            self._create_legacy_pack(pack)

            scorer = StubScorer(
                {
                    "p1": [0.5, 0.5, 0.5, 0.5, 0.5],
                    "p2": [0.9, 0.9, 0.9, 0.9, 0.9],
                }
            )

            report1 = run_rating_job(
                input_path=pack,
                output_path=out,
                scorer=scorer,
                report_path=report_path,
                batch_size=1,
                limit=1,
                progress_every=0,
            )
            self.assertEqual(report1["totalPackPuzzles"], 3)
            self.assertEqual(report1["totalArrowDuelEligible"], 2)
            self.assertEqual(report1["ratedCount"], 1)

            report2 = run_rating_job(
                input_path=pack,
                output_path=out,
                scorer=scorer,
                report_path=report_path,
                batch_size=2,
                resume=True,
                progress_every=0,
            )
            self.assertEqual(report2["ratedCount"], 2)
            saved_report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_report["ratedCount"], 2)

            conn = sqlite3.connect(out)
            by_id = {
                r[0]: (r[1], r[2])
                for r in conn.execute("SELECT puzzle_id, original_rating, arrow_duel_rating FROM arrow_duel_ratings")
            }
            conn.close()
            self.assertEqual(by_id["p1"], (1500, 1500))
            self.assertEqual(by_id["p2"], (2100, 600))


if __name__ == "__main__":
    unittest.main()
