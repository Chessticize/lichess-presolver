from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from classify_arrow_duel_routes import (  # noqa: E402
    RouteConfig,
    decode_pack_move,
    decode_pack_position,
    decode_compact_metric,
    encode_compact_metric,
    is_competitive,
)
from presolve_lichess_puzzles import Evaluation  # noqa: E402


ROUTE_ROWS = [
    (
        "RVpOb",
        "r1br2k1/6qp/p2p1pp1/1ppN4/2PnP1P1/5P2/PP4PQ/1BKR3R w - - 4 22",
        "h2f4 d4e2 c1c2 e2f4",
        858,
        "h2h7",
    ),
    (
        "02ARe",
        "rq3rk1/2p2ppp/4pn2/p2pP3/b7/1PP1P1P1/P1QN1PP1/2KR3R b - -",
        "f6d7 c2h7",
        674,
        "a4b3",
    ),
    (
        "00SMl",
        "r4rk1/ppp2pn1/2np4/q2N4/3PP3/5P2/PPP5/1K1R1B1R b - -",
        "c6b4 d5b4",
        799,
        "a5d5",
    ),
    (
        "01N0w",
        "2kr3r/1pp3Q1/p1p5/2b2p1P/q1P5/6BP/PP3P2/1K1RR3 w - -",
        "g7f6 d8d1 e1d1 a4d1",
        990,
        "g7c7",
    ),
    (
        "promotion-excluded",
        "7k/P7/8/8/8/8/8/7K w - - 0 1",
        "a7a8r",
        900,
        "a7a8q",
    ),
]

PIECE_CODES = {
    piece: code
    for code, piece in enumerate(
        (None, "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")
    )
    if piece is not None
}
PROMOTION_CODES = {"q": 1, "r": 2, "b": 3, "n": 4}


def square_index(square: str) -> int:
    return ord(square[0]) - ord("a") + (int(square[1]) - 1) * 8


def encode_move(move: str) -> bytes:
    encoded = (
        square_index(move[:2])
        | (square_index(move[2:4]) << 6)
        | (PROMOTION_CODES.get(move[4:], 0) << 12)
    )
    return bytes((encoded & 0xFF, encoded >> 8))


def encode_position(fen: str) -> bytes:
    board, side, castling, en_passant, *_ = fen.split()
    pieces: list[str | None] = [None] * 64
    for rank_from_eight, rank_text in enumerate(board.split("/")):
        file_index = 0
        for symbol in rank_text:
            if symbol.isdigit():
                file_index += int(symbol)
            else:
                pieces[(7 - rank_from_eight) * 8 + file_index] = symbol
                file_index += 1
    occupied = [index for index, piece in enumerate(pieces) if piece is not None]
    result = bytearray(10 + (len(occupied) + 1) // 2)
    for square in occupied:
        result[square >> 3] |= 1 << (square & 7)
    metadata = int(side == "b")
    for bit, symbol in ((1, "K"), (2, "Q"), (3, "k"), (4, "q")):
        if symbol in castling:
            metadata |= 1 << bit
    if en_passant != "-":
        metadata |= (square_index(en_passant) + 1) << 5
    result[8] = metadata & 0xFF
    result[9] = metadata >> 8
    for index, square in enumerate(occupied):
        code = PIECE_CODES[pieces[square]]
        result[10 + (index >> 1)] |= code if index % 2 == 0 else code << 4
    return bytes(result)


def binary_route_row(row: tuple[object, ...]) -> tuple[object, ...]:
    puzzle_id, fen, solution_moves, rating, bestmove = row
    assert isinstance(fen, str)
    assert isinstance(solution_moves, str)
    assert isinstance(bestmove, str)
    return (
        puzzle_id,
        encode_position(fen),
        b"".join(encode_move(move) for move in solution_moves.split()),
        rating,
        encode_move(bestmove),
    )


class CompetitiveMoveTests(unittest.TestCase):
    def test_route_contract_uses_only_best_and_second_best(self) -> None:
        self.assertEqual(RouteConfig().multipv, 2)
        with self.assertRaisesRegex(ValueError, "multipv must be 2"):
            RouteConfig(multipv=5)

    def test_decodes_the_versioned_binary_core_pack_profile(self) -> None:
        start_position = bytes(
            [
                0xFF,
                0xFF,
                0x00,
                0x00,
                0x00,
                0x00,
                0xFF,
                0xFF,
                0x1E,
                0x00,
                0x24,
                0x53,
                0x36,
                0x42,
                0x11,
                0x11,
                0x11,
                0x11,
                0x77,
                0x77,
                0x77,
                0x77,
                0x8A,
                0xB9,
                0x9C,
                0xA8,
            ]
        )

        self.assertEqual(
            decode_pack_position(start_position),
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        )
        self.assertEqual(decode_pack_move(bytes([0x0C, 0x07])), "e2e4")

    def test_compact_metric_round_trip_preserves_all_public_codes(self) -> None:
        values = [None, 1, 4, 6, 11, 15]

        for value in values:
            with self.subTest(value=value):
                self.assertEqual(
                    decode_compact_metric(encode_compact_metric(value)), value
                )

        with self.assertRaises(ValueError):
            decode_compact_metric(16)

    def test_uses_centipawn_tolerance_from_the_initial_players_view(self) -> None:
        fen = "8/8/8/8/8/8/8/K6k b - - 0 1"
        best = Evaluation(score_cp=-200, bestmove="h1h2")

        self.assertTrue(
            is_competitive(
                best,
                Evaluation(score_cp=-120, bestmove="h1g1"),
                fen=fen,
                tolerance_cp=100,
            )
        )
        self.assertFalse(
            is_competitive(
                best,
                Evaluation(score_cp=-50, bestmove="h1g2"),
                fen=fen,
                tolerance_cp=100,
            )
        )

    def test_only_treats_equal_mate_distance_as_an_equivalent_best_move(self) -> None:
        fen = "8/8/8/8/8/8/8/K6k w - - 0 1"
        best = Evaluation(score_cp=10000, bestmove="a1a2", mate_in=1)

        self.assertTrue(
            is_competitive(
                best,
                Evaluation(score_cp=10000, bestmove="a1b1", mate_in=1),
                fen=fen,
                tolerance_cp=100,
            )
        )
        self.assertFalse(
            is_competitive(
                best,
                Evaluation(score_cp=10000, bestmove="a1b2", mate_in=2),
                fen=fen,
                tolerance_cp=100,
            )
        )


@unittest.skipUnless(shutil.which("stockfish"), "Stockfish is not installed")
class RouteCliTests(unittest.TestCase):
    def test_writes_route_depth_independent_of_worker_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pack = root / "core-pack.sqlite"
            binary_pack = root / "binary-core-pack.sqlite"
            first_output = root / "first.sqlite"
            second_output = root / "second.sqlite"
            compact_output = root / "compact.sqlite"
            report = root / "report.json"
            connection = sqlite3.connect(pack)
            connection.execute(
                """
                CREATE TABLE puzzles (
                  id TEXT PRIMARY KEY,
                  initial_fen TEXT NOT NULL,
                  solution_moves TEXT NOT NULL,
                  rating INTEGER NOT NULL,
                  stockfish_bestmove TEXT NOT NULL
                )
                """
            )
            connection.executemany(
                "INSERT INTO puzzles VALUES (?, ?, ?, ?, ?)",
                ROUTE_ROWS,
            )
            connection.commit()
            connection.close()

            connection = sqlite3.connect(binary_pack)
            connection.executescript(
                """
                CREATE TABLE puzzles (
                  id TEXT PRIMARY KEY,
                  initial_fen BLOB NOT NULL,
                  solution_moves BLOB NOT NULL,
                  rating INTEGER NOT NULL,
                  stockfish_bestmove BLOB NOT NULL
                );
                CREATE TABLE pack_format (
                  id INTEGER PRIMARY KEY,
                  format_id TEXT NOT NULL,
                  pack_schema_version INTEGER NOT NULL,
                  position_codec TEXT NOT NULL,
                  position_codec_version INTEGER NOT NULL,
                  move_codec TEXT NOT NULL,
                  move_codec_version INTEGER NOT NULL
                );
                INSERT INTO pack_format VALUES (
                  1,
                  'chessticize-core-pack',
                  2,
                  'chessticize-position',
                  1,
                  'chessticize-uci16',
                  1
                );
                """
            )
            connection.executemany(
                "INSERT INTO puzzles VALUES (?, ?, ?, ?, ?)",
                [binary_route_row(row) for row in ROUTE_ROWS],
            )
            connection.commit()
            connection.close()

            base_command = [
                sys.executable,
                str(REPOSITORY_ROOT / "classify_arrow_duel_routes.py"),
                str(pack),
            ]
            first = subprocess.run(
                [
                    *base_command,
                    str(first_output),
                    "--report",
                    str(report),
                    "--compact-output",
                    str(compact_output),
                    "--depth",
                    "10",
                    "--jobs",
                    "1",
                    "--batch-size",
                    "4",
                    "--progress-every",
                    "0",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            second = subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "classify_arrow_duel_routes.py"),
                    str(binary_pack),
                    str(second_output),
                    "--depth",
                    "10",
                    "--jobs",
                    "2",
                    "--batch-size",
                    "1",
                    "--progress-every",
                    "0",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(second.returncode, 0, second.stderr)

            first_db = sqlite3.connect(first_output)
            first_rows = first_db.execute(
                "SELECT * FROM arrow_duel_route_metrics ORDER BY puzzle_id"
            ).fetchall()
            definitions = first_db.execute(
                "SELECT id, key FROM arrow_duel_metric_definitions ORDER BY id"
            ).fetchall()
            with self.assertRaises(sqlite3.IntegrityError):
                first_db.execute(
                    "INSERT INTO arrow_duel_route_metrics VALUES ('bad', 0)"
                )
            first_db.close()

            second_db = sqlite3.connect(second_output)
            second_rows = second_db.execute(
                "SELECT * FROM arrow_duel_route_metrics ORDER BY puzzle_id"
            ).fetchall()
            second_db.close()

            compact_db = sqlite3.connect(compact_output)
            compact_rows = compact_db.execute(
                """
                SELECT first_puzzle_rowid, puzzle_count, packed_metrics
                FROM arrow_duel_route_metric_chunks
                ORDER BY chunk_id
                """
            ).fetchall()
            compact_definitions = compact_db.execute(
                """
                SELECT id, key, bit_width
                FROM arrow_duel_metric_definitions
                ORDER BY id
                """
            ).fetchall()
            compact_db.close()

            self.assertEqual(first_rows, second_rows)
            self.assertEqual(
                first_rows,
                [
                    ("00SMl", 4),
                    ("01N0w", 1),
                    ("02ARe", 11),
                    ("RVpOb", 6),
                ],
            )
            self.assertEqual(
                definitions,
                [
                    (1, "top_move_unique_plies"),
                    (2, "unique_followup_moves"),
                ],
            )
            self.assertEqual(compact_rows, [(1, 5, bytes([182, 20, 0]))])
            self.assertEqual(
                compact_definitions,
                [(1, "top_move_unique_plies", 4)],
            )

            summary = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(summary["input"]["routeMetrics"]["multipv"], 2)
            self.assertEqual(summary["packPuzzleCount"], 5)
            self.assertEqual(summary["arrowDuelInputCount"], 4)
            self.assertEqual(
                summary["topMoveUniquePliesDistribution"],
                {"1": 1, "4": 1, "6": 1, "11": 1},
            )
            self.assertEqual(
                summary["uniqueFollowupMovesDistribution"],
                {"0": 1, "1": 1, "2": 1, "4+": 1},
            )
            self.assertEqual(
                set(summary["samplesByUniqueFollowupMoves"]),
                {"0", "1", "2", "3", "4+"},
            )
            self.assertEqual(
                {
                    key: [sample["puzzleId"] for sample in samples]
                    for key, samples in summary[
                        "samplesByUniqueFollowupMoves"
                    ].items()
                },
                {
                    "0": ["01N0w"],
                    "1": ["00SMl"],
                    "2": ["RVpOb"],
                    "3": [],
                    "4+": ["02ARe"],
                },
            )
            self.assertEqual(summary["compactOutput"]["payloadBytes"], 3)
            self.assertEqual(summary["compactOutput"]["metricCount"], 4)


if __name__ == "__main__":
    unittest.main()
