#!/usr/bin/env python3
"""Compute Maia-backed Arrow Duel difficulty ratings for a Chessticize Core Pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from classify_arrow_duel_routes import (
    PACK_PIECES,
    arrow_duel_input_sql,
    decode_pack_move,
    decode_pack_position,
    normalize_uci,
    validate_pack,
)

SCHEMA_VERSION = 1
GLICKO_SCALE = 400.0 / math.log(10.0)  # 173.71779276130073
DEFAULT_ELO_GRID: tuple[int, ...] = (700, 1100, 1500, 1900, 2300)
DEFAULT_MIN_RATING = 600
DEFAULT_MAX_RATING = 2200
DEFAULT_TARGET_PROB = 0.75
DEFAULT_FULL_CHECK_RATE = 1.0
DEFAULT_SPRINT_CHECK_RATE = 0.75

# Piece letter -> Maia-3 channel index (0..11) for White-to-move orientation
_FEN_PIECE_TO_CHANNEL: dict[str, int] = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}


@dataclass(frozen=True)
class ArrowDuelPuzzleRow:
    puzzle_id: str
    rating: int
    fen_raw: bytes | str
    bestmove_raw: bytes | str
    blunder_raw: bytes | str


@dataclass(frozen=True)
class ArrowDuelRatingResult:
    puzzle_id: str
    original_rating: int
    arrow_duel_rating: int
    arrow_duel_rating_sprint: int
    q_by_elo_bp: tuple[int, ...]  # basis points 0..10000 for each Elo in grid


class PairwisePolicyScorer(Protocol):
    model_identity: dict[str, object]

    def score_batch(
        self,
        rows: Sequence[ArrowDuelPuzzleRow],
        row_encoding: str,
        elos: Sequence[int],
    ) -> np.ndarray:
        """Return float32 array of shape (len(rows), len(elos)) with q(E) in [0, 1]."""
        ...


def sigmoid_scalar(x: float) -> float:
    if x >= 35.0:
        return 1.0
    if x <= -35.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def solve_prompted_arrow_duel_rating(
    original_rating: float,
    elos: Sequence[int],
    qs: Sequence[float],
    *,
    target_prob: float = DEFAULT_TARGET_PROB,
    check_rate: float = DEFAULT_FULL_CHECK_RATE,
    min_rating: int = DEFAULT_MIN_RATING,
    max_rating: int = DEFAULT_MAX_RATING,
) -> int:
    """
    Solve P_AD(E*) = target_prob for the first-principles prompted-comparison model:

        p_refute(E) = check_rate * sigmoid((E - original_rating) / 173.7178)
        P_AD(E)     = p_refute(E) + (1 - p_refute(E)) * q_maia(E)

    When q_maia(E) = 0.5 and check_rate = 1.0, P_AD(original_rating) = 0.75,
    matching the 3PL-IRT (c = 0.5) difficulty definition on the exact Lichess Elo scale.
    """
    if len(elos) != len(qs) or not elos:
        raise ValueError("elos and qs must have identical non-zero length")

    # Vectorized search on a 5-Elo grid from 400 to 2800, then linear interpolation
    elo_arr = np.asarray(elos, dtype=np.float64)
    q_arr = np.clip(np.asarray(qs, dtype=np.float64), 0.0, 1.0)

    lo_grid = 400.0
    hi_grid = 2800.0
    step = 5.0
    grid = np.arange(lo_grid, hi_grid + 1.0, step, dtype=np.float64)
    q_grid = np.interp(grid, elo_arr, q_arr)
    z = np.clip((grid - float(original_rating)) / GLICKO_SCALE, -35.0, 35.0)
    p_ref = check_rate / (1.0 + np.exp(-z))
    p_ad = p_ref + (1.0 - p_ref) * q_grid

    if p_ad[0] >= target_prob:
        return int(min_rating)
    if p_ad[-1] <= target_prob:
        return int(max_rating)

    idx = int(np.argmax(p_ad >= target_prob))
    if idx == 0:
        est = float(grid[0])
    else:
        e0, e1 = float(grid[idx - 1]), float(grid[idx])
        p0, p1 = float(p_ad[idx - 1]), float(p_ad[idx])
        frac = (target_prob - p0) / max(1e-9, p1 - p0)
        est = e0 + frac * (e1 - e0)
    return int(max(min_rating, min(max_rating, round(est))))


def solve_prompted_arrow_duel_batch(
    original_ratings: np.ndarray,
    elos: Sequence[int],
    q_matrix: np.ndarray,
    *,
    target_prob: float = DEFAULT_TARGET_PROB,
    check_rate: float = DEFAULT_FULL_CHECK_RATE,
    min_rating: int = DEFAULT_MIN_RATING,
    max_rating: int = DEFAULT_MAX_RATING,
) -> np.ndarray:
    """Vectorized version of solve_prompted_arrow_duel_rating for a batch (N, K)."""
    n = len(original_ratings)
    if n == 0:
        return np.empty((0,), dtype=np.int32)

    elo_arr = np.asarray(elos, dtype=np.float32)
    grid = np.arange(400.0, 2801.0, 5.0, dtype=np.float32)  # (G,)
    g_len = len(grid)

    # Piecewise linear interpolation of q_matrix (N, K) onto grid (G,)
    q_clip = np.clip(q_matrix.astype(np.float32), 0.0, 1.0)
    idx_hi = np.searchsorted(elo_arr, grid, side="right")
    idx_hi = np.clip(idx_hi, 1, len(elo_arr) - 1)
    idx_lo = idx_hi - 1
    w = np.clip(
        (grid - elo_arr[idx_lo]) / np.maximum(1e-6, elo_arr[idx_hi] - elo_arr[idx_lo]),
        0.0,
        1.0,
    )  # (G,)
    # q_grid: (N, G)
    q_grid = q_clip[:, idx_lo] * (1.0 - w[None, :]) + q_clip[:, idx_hi] * w[None, :]

    z = np.clip(
        (grid[None, :] - original_ratings.astype(np.float32)[:, None]) / np.float32(GLICKO_SCALE),
        -35.0,
        35.0,
    )
    p_ref = np.float32(check_rate) / (1.0 + np.exp(-z))
    p_ad = p_ref + (1.0 - p_ref) * q_grid

    reached = p_ad >= np.float32(target_prob)
    first_idx = np.argmax(reached, axis=1)  # 0 if never reached OR reached at index 0
    any_reached = reached.any(axis=1)
    at_start = reached[:, 0]

    # Linear interpolation for interior crossings
    prev_idx = np.maximum(0, first_idx - 1)
    row_arange = np.arange(n)
    p0 = p_ad[row_arange, prev_idx]
    p1 = p_ad[row_arange, first_idx]
    e0 = grid[prev_idx]
    e1 = grid[first_idx]
    frac = (np.float32(target_prob) - p0) / np.maximum(np.float32(1e-9), p1 - p0)
    est = e0 + frac * (e1 - e0)

    est = np.where(at_start, np.float32(min_rating), est)
    est = np.where(~any_reached, np.float32(max_rating), est)
    return np.clip(np.rint(est), min_rating, max_rating).astype(np.int32)


def encode_binary_v1_position_to_maia3(blob: bytes, out_64x12: np.ndarray) -> bool:
    """
    Fill a zeroed (64, 12) float32 array directly from a Core Pack binary-v1 position BLOB,
    matching `maia3.dataset.tokenize_board(chess.Board(fen))` bit-for-bit.
    Returns True if Black is to move (meaning moves must be rank-mirrored).
    """
    metadata = blob[8] | (blob[9] << 8)
    is_black = bool(metadata & 1)
    piece_index = 0
    for byte_idx in range(8):
        b = blob[byte_idx]
        if not b:
            continue
        base_sq = byte_idx << 3
        while b:
            lsb = b & -b
            sq = base_sq + lsb.bit_length() - 1
            b ^= lsb
            encoded_piece = blob[10 + (piece_index >> 1)]
            code = (encoded_piece & 0x0F) if (piece_index & 1) == 0 else (encoded_piece >> 4)
            piece_index += 1
            if is_black:
                target_sq = sq ^ 56
                ch = code + 5 if code <= 6 else code - 7
            else:
                target_sq = sq
                ch = code - 1
            out_64x12[target_sq, ch] = 1.0
    return is_black


def encode_fen_to_maia3(fen: str, out_64x12: np.ndarray) -> bool:
    """Fill a zeroed (64, 12) float32 array directly from a FEN string."""
    fields = fen.strip().split()
    placement = fields[0]
    is_black = len(fields) > 1 and fields[1] == "b"
    rank = 7
    file_idx = 0
    for ch_char in placement:
        if ch_char == "/":
            rank -= 1
            file_idx = 0
        elif "1" <= ch_char <= "8":
            file_idx += ord(ch_char) - ord("0")
        else:
            sq = (rank << 3) | file_idx
            file_idx += 1
            ch = _FEN_PIECE_TO_CHANNEL[ch_char]
            if is_black:
                sq ^= 56
                ch = ch + 6 if ch < 6 else ch - 6
            out_64x12[sq, ch] = 1.0
    return is_black


def decode_binary_v1_move_squares(blob2: bytes) -> tuple[int, int]:
    encoded = blob2[0] | (blob2[1] << 8)
    return encoded & 0x3F, (encoded >> 6) & 0x3F


def decode_uci_move_squares(uci: str) -> tuple[int, int]:
    u = normalize_uci(uci)
    from_sq = (ord(u[0]) - 97) | ((ord(u[1]) - 49) << 3)
    to_sq = (ord(u[2]) - 97) | ((ord(u[3]) - 49) << 3)
    return from_sq, to_sq


class Maia3PairwiseScorer:
    """Batched GPU/MPS/CPU Maia-3 pairwise Arrow Duel scorer."""

    def __init__(self, model_alias: str = "5m", device: str = "auto") -> None:
        import torch
        from maia3.model_registry import (
            apply_model_config,
            resolve_checkpoint_path,
            resolve_model_spec,
        )
        from maia3.uci import load_model
        from maia3.utils import get_all_possible_moves

        self.torch = torch
        if device == "auto":
            if torch.cuda.is_available():
                resolved_device = "cuda"
            elif torch.backends.mps.is_available():
                resolved_device = "mps"
            else:
                resolved_device = "cpu"
        else:
            resolved_device = device
        self.device = resolved_device

        spec = resolve_model_spec(model_alias)
        ckpt_path = resolve_checkpoint_path(spec)
        self.cfg = argparse.Namespace(
            device=resolved_device,
            checkpoint_path=ckpt_path,
            use_amp=False,
            trust_checkpoint=False,
            use_uci_history=False,
        )
        apply_model_config(self.cfg, spec)
        self.model = load_model(self.cfg)
        self.model.eval()

        # Precompute (64, 64) non-promotion move lookup table -> Maia-3 move vocab index
        all_moves = get_all_possible_moves()
        move_to_idx = {m: i for i, m in enumerate(all_moves)}
        lut = np.full((64, 64), -1, dtype=np.int64)
        for f_sq in range(64):
            f_name = f"{chr(97 + (f_sq & 7))}{(f_sq >> 3) + 1}"
            for t_sq in range(64):
                if f_sq == t_sq:
                    continue
                t_name = f"{chr(97 + (t_sq & 7))}{(t_sq >> 3) + 1}"
                idx = move_to_idx.get(f_name + t_name)
                if idx is not None:
                    lut[f_sq, t_sq] = idx
        self.move_lut = lut
        self.history_len = int(self.cfg.history)
        self.include_time_info = bool(self.cfg.include_time_info)
        self.model_identity = {
            "family": "maia3",
            "name": spec.name,
            "repoId": spec.repo_id,
            "checkpointFilename": spec.checkpoint_filename,
            "checkpointSha256": f"sha256:{sha256_file(Path(ckpt_path))}",
            "device": resolved_device,
            "parameterCount": int(sum(p.numel() for p in self.model.parameters())),
        }

    def score_batch(
        self,
        rows: Sequence[ArrowDuelPuzzleRow],
        row_encoding: str,
        elos: Sequence[int],
    ) -> np.ndarray:
        torch = self.torch
        n = len(rows)
        k = len(elos)
        board_arr = np.zeros((n, 64, 12), dtype=np.float32)
        good_idx = np.empty((n,), dtype=np.int64)
        bad_idx = np.empty((n,), dtype=np.int64)
        lut = self.move_lut

        if row_encoding == "binary-v1":
            for i, row in enumerate(rows):
                assert isinstance(row.fen_raw, bytes)
                assert isinstance(row.bestmove_raw, bytes)
                assert isinstance(row.blunder_raw, bytes)
                is_black = encode_binary_v1_position_to_maia3(row.fen_raw, board_arr[i])
                gf, gt = decode_binary_v1_move_squares(row.bestmove_raw)
                bf, bt = decode_binary_v1_move_squares(row.blunder_raw)
                if is_black:
                    gf ^= 56
                    gt ^= 56
                    bf ^= 56
                    bt ^= 56
                g_i = lut[gf, gt]
                b_i = lut[bf, bt]
                if g_i < 0 or b_i < 0:
                    raise ValueError(
                        f"Puzzle {row.puzzle_id!r} has move outside Maia-3 non-promotion vocabulary"
                    )
                good_idx[i] = g_i
                bad_idx[i] = b_i
        else:
            for i, row in enumerate(rows):
                assert isinstance(row.fen_raw, str)
                assert isinstance(row.bestmove_raw, str)
                assert isinstance(row.blunder_raw, str)
                is_black = encode_fen_to_maia3(row.fen_raw, board_arr[i])
                gf, gt = decode_uci_move_squares(row.bestmove_raw)
                bf, bt = decode_uci_move_squares(row.blunder_raw)
                if is_black:
                    gf ^= 56
                    gt ^= 56
                    bf ^= 56
                    bt ^= 56
                g_i = lut[gf, gt]
                b_i = lut[bf, bt]
                if g_i < 0 or b_i < 0:
                    raise ValueError(
                        f"Puzzle {row.puzzle_id!r} has move outside Maia-3 non-promotion vocabulary"
                    )
                good_idx[i] = g_i
                bad_idx[i] = b_i

        # Build (N, 64, 12*history + time_cols) tensor on target device once for all Elos
        extra_cols = 4 if self.include_time_info else 1
        with torch.no_grad():
            base_t = torch.from_numpy(board_arr).to(self.device)
            hist_t = base_t.repeat(1, 1, self.history_len)
            zeros_t = torch.zeros((n, 64, extra_cols), dtype=torch.float32, device=self.device)
            tokens = torch.cat([hist_t, zeros_t], dim=2)  # (N, 64, 97)

            row_idx = torch.arange(n, dtype=torch.long, device=self.device)
            good_t = torch.from_numpy(good_idx).to(self.device)
            bad_t = torch.from_numpy(bad_idx).to(self.device)

            q_out = np.empty((n, k), dtype=np.float32)
            for e_idx, elo in enumerate(elos):
                elo_t = torch.full((n,), int(elo), dtype=torch.long, device=self.device)
                logits_move, _, _ = self.model(tokens, elo_t, elo_t)
                logits_f = logits_move.float()
                diff = logits_f[row_idx, good_t] - logits_f[row_idx, bad_t]
                q_out[:, e_idx] = torch.sigmoid(diff).cpu().numpy()

        return q_out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_identity(
    *,
    input_path: Path,
    input_sha256: str,
    row_encoding: str,
    model_identity: dict[str, object],
    elos: Sequence[int],
    min_rating: int,
    max_rating: int,
    target_prob: float,
    full_check_rate: float,
    sprint_check_rate: float,
) -> dict[str, object]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "inputPath": str(input_path.resolve()),
        "inputSha256": f"sha256:{input_sha256}",
        "inputRowEncoding": row_encoding,
        "model": model_identity,
        "formula": {
            "name": "prompted-comparison-3pl-irt-v1",
            "eloGrid": list(elos),
            "targetProb": target_prob,
            "fullCheckRate": full_check_rate,
            "sprintCheckRate": sprint_check_rate,
            "minRating": min_rating,
            "maxRating": max_rating,
            "glickoScale": GLICKO_SCALE,
        },
    }


def initialize_output_db(
    path: Path,
    identity: dict[str, object],
    *,
    overwrite: bool,
    resume: bool,
) -> tuple[sqlite3.Connection, int]:
    if path.exists() and overwrite:
        path.unlink()
    if path.exists() and not resume:
        raise FileExistsError(f"Output already exists: {path}; pass --resume or --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS analysis_metadata (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS arrow_duel_ratings (
          puzzle_id TEXT PRIMARY KEY,
          original_rating INTEGER NOT NULL,
          arrow_duel_rating INTEGER NOT NULL,
          arrow_duel_rating_sprint INTEGER NOT NULL,
          q_700_bp INTEGER NOT NULL,
          q_1100_bp INTEGER NOT NULL,
          q_1500_bp INTEGER NOT NULL,
          q_1900_bp INTEGER NOT NULL,
          q_2300_bp INTEGER NOT NULL
        ) WITHOUT ROWID;
        """
    )

    existing_identity = conn.execute(
        "SELECT value FROM analysis_metadata WHERE key = 'identity'"
    ).fetchone()
    identity_json = json.dumps(identity, sort_keys=True)
    if existing_identity is None:
        conn.execute(
            "INSERT INTO analysis_metadata (key, value) VALUES ('identity', ?)",
            (identity_json,),
        )
        conn.execute(
            "INSERT OR REPLACE INTO analysis_metadata (key, value) VALUES ('createdAt', ?)",
            (datetime.now(timezone.utc).isoformat(),),
        )
        conn.commit()
    else:
        prior = json.loads(existing_identity[0])
        comparable_prior = {k: v for k, v in prior.items() if k != "inputPath"}
        comparable_now = {k: v for k, v in identity.items() if k != "inputPath"}
        if comparable_prior != comparable_now:
            conn.close()
            raise ValueError(
                "Cannot --resume output database: analysis identity mismatch "
                "(input hash, model, or formula differs)"
            )

    already_done = int(conn.execute("SELECT COUNT(*) FROM arrow_duel_ratings").fetchone()[0])
    return conn, already_done


def iter_pending_arrow_duel_rows(
    src_conn: sqlite3.Connection,
    out_conn: sqlite3.Connection,
    row_encoding: str,
    *,
    batch_size: int,
    limit: int | None,
):
    done_ids: set[str] = set()
    for (pid,) in out_conn.execute("SELECT puzzle_id FROM arrow_duel_ratings"):
        done_ids.add(pid)

    where_sql = arrow_duel_input_sql(row_encoding)
    if row_encoding == "binary-v1":
        blunder_expr = "substr(p.solution_moves, 1, 2)"
    else:
        blunder_expr = """
          CASE
            WHEN instr(trim(p.solution_moves), ' ') = 0 THEN trim(p.solution_moves)
            ELSE substr(trim(p.solution_moves), 1, instr(trim(p.solution_moves), ' ') - 1)
          END
        """

    sql = f"""
        SELECT p.id, p.rating, p.initial_fen, p.stockfish_bestmove, {blunder_expr}
        FROM puzzles p
        WHERE {where_sql}
        ORDER BY p.id
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    cursor = src_conn.execute(sql)
    batch: list[ArrowDuelPuzzleRow] = []
    for pid, rating, fen_raw, bm_raw, blunder_raw in cursor:
        if pid in done_ids:
            continue
        batch.append(ArrowDuelPuzzleRow(pid, int(rating), fen_raw, bm_raw, blunder_raw))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_report(
    out_conn: sqlite3.Connection,
    identity: dict[str, object],
    *,
    total_pack_puzzles: int,
    total_eligible: int,
    elapsed_seconds: float,
) -> dict[str, object]:
    rows = out_conn.execute(
        """
        SELECT puzzle_id, original_rating, arrow_duel_rating, arrow_duel_rating_sprint, q_1500_bp
        FROM arrow_duel_ratings
        ORDER BY puzzle_id
        """
    ).fetchall()
    n = len(rows)
    if n == 0:
        return {
            "identity": identity,
            "totalPackPuzzles": total_pack_puzzles,
            "totalArrowDuelEligible": total_eligible,
            "ratedCount": 0,
            "elapsedSeconds": round(elapsed_seconds, 2),
        }

    orig = np.fromiter((r[1] for r in rows), dtype=np.float64, count=n)
    ad = np.fromiter((r[2] for r in rows), dtype=np.float64, count=n)
    ad_s = np.fromiter((r[3] for r in rows), dtype=np.float64, count=n)
    q15 = np.fromiter((r[4] / 10000.0 for r in rows), dtype=np.float64, count=n)

    def summarize_pair(arr: np.ndarray) -> dict[str, object]:
        diff = arr - orig
        abs_diff = np.abs(diff)
        r_val = (
            float(np.corrcoef(orig, arr)[0, 1])
            if n > 1 and np.std(orig) > 0 and np.std(arr) > 0
            else 0.0
        )
        # Spearman rank correlation via argsort ranks (no scipy dependency required)
        if n > 1 and np.std(orig) > 0 and np.std(arr) > 0:
            orig_ranks = np.empty(n, dtype=np.float64)
            orig_ranks[np.argsort(orig, kind="mergesort")] = np.arange(n, dtype=np.float64)
            arr_ranks = np.empty(n, dtype=np.float64)
            arr_ranks[np.argsort(arr, kind="mergesort")] = np.arange(n, dtype=np.float64)
            rho_val = float(np.corrcoef(orig_ranks, arr_ranks)[0, 1])
        else:
            rho_val = 0.0
        return {
            "mean": round(float(arr.mean()), 2),
            "std": round(float(arr.std()), 2),
            "min": int(arr.min()),
            "max": int(arr.max()),
            "pearsonR": round(r_val, 4),
            "spearmanRho": round(rho_val, 4),
            "deltaMean": round(float(diff.mean()), 2),
            "deltaMedian": round(float(np.median(diff)), 2),
            "deltaMad": round(float(np.median(abs_diff)), 2),
            "shareWithin150": round(float((abs_diff <= 150).mean()), 4),
            "shareWithin300": round(float((abs_diff <= 300).mean()), 4),
            "shareDropAtLeast400": round(float((diff <= -400).mean()), 4),
            "shareRiseAtLeast400": round(float((diff >= 400).mean()), 4),
        }

    orig_buckets = []
    for lo in range(600, 2200, 200):
        hi = lo + 200 if lo < 2000 else 2201
        mask = (orig >= lo) & (orig < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        sub = ad[mask]
        p10, p50, p90 = np.percentile(sub, [10, 50, 90])
        orig_buckets.append(
            {
                "minOriginalRating": lo,
                "maxOriginalRating": 2200 if lo == 2000 else lo + 199,
                "count": cnt,
                "adMean": round(float(sub.mean()), 1),
                "adP10": int(round(float(p10))),
                "adMedian": int(round(float(p50))),
                "adP90": int(round(float(p90))),
                "shareTrivialLe900": round(float((sub <= 900).mean()), 4),
            }
        )

    ad_buckets = []
    for lo in range(600, 2200, 100):
        hi = lo + 100 if lo < 2100 else 2201
        cnt = int(((ad >= lo) & (ad < hi)).sum())
        ad_buckets.append(
            {
                "minRating": lo,
                "maxRating": 2200 if lo == 2100 else lo + 99,
                "count": cnt,
            }
        )

    # Top downward & upward shift sample IDs
    diffs = ad - orig
    down_idx = np.argsort(diffs)[:10]
    up_idx = np.argsort(-diffs)[:10]

    def sample_entry(i: int) -> dict[str, object]:
        return {
            "puzzleId": rows[i][0],
            "originalRating": int(rows[i][1]),
            "arrowDuelRating": int(rows[i][2]),
            "delta": int(rows[i][2] - rows[i][1]),
            "q1500": round(rows[i][4] / 10000.0, 4),
        }

    return {
        "identity": identity,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "totalPackPuzzles": total_pack_puzzles,
        "totalArrowDuelEligible": total_eligible,
        "ratedCount": n,
        "elapsedSeconds": round(elapsed_seconds, 2),
        "maiaIntuitionAt1500": {
            "shareBlunderMoreTemptingLt35": round(float((q15 < 0.35).mean()), 4),
            "shareBalanced35To65": round(float(((q15 >= 0.35) & (q15 <= 0.65)).mean()), 4),
            "shareGoodObviousGt65": round(float((q15 > 0.65).mean()), 4),
            "shareGoodOverwhelmingGt85": round(float((q15 > 0.85).mean()), 4),
        },
        "arrowDuelRating": summarize_pair(ad),
        "arrowDuelRatingSprint": summarize_pair(ad_s),
        "byOriginalBucket": orig_buckets,
        "arrowDuelBuckets": ad_buckets,
        "sampleLargestDrops": [sample_entry(int(i)) for i in down_idx],
        "sampleLargestRises": [sample_entry(int(i)) for i in up_idx],
    }


def run_rating_job(
    *,
    input_path: Path,
    output_path: Path,
    scorer: PairwisePolicyScorer,
    report_path: Path | None = None,
    elos: Sequence[int] = DEFAULT_ELO_GRID,
    batch_size: int = 1024,
    limit: int | None = None,
    overwrite: bool = False,
    resume: bool = False,
    progress_every: int = 50_000,
) -> dict[str, object]:
    if tuple(elos) != DEFAULT_ELO_GRID:
        raise ValueError(f"Expected canonical Elo grid {DEFAULT_ELO_GRID}, got {tuple(elos)}")

    src_conn = sqlite3.connect(f"file:{input_path}?mode=ro", uri=True)
    try:
        layout = validate_pack(src_conn)
        where_sql = arrow_duel_input_sql(layout.row_encoding)
        total_eligible = int(
            src_conn.execute(f"SELECT COUNT(*) FROM puzzles p WHERE {where_sql}").fetchone()[0]
        )
        effective_total = min(total_eligible, limit) if limit is not None else total_eligible
        input_sha256 = sha256_file(input_path)
        identity = build_identity(
            input_path=input_path,
            input_sha256=input_sha256,
            row_encoding=layout.row_encoding,
            model_identity=scorer.model_identity,
            elos=elos,
            min_rating=DEFAULT_MIN_RATING,
            max_rating=DEFAULT_MAX_RATING,
            target_prob=DEFAULT_TARGET_PROB,
            full_check_rate=DEFAULT_FULL_CHECK_RATE,
            sprint_check_rate=DEFAULT_SPRINT_CHECK_RATE,
        )

        out_conn, already_done = initialize_output_db(
            output_path, identity, overwrite=overwrite, resume=resume
        )
        t0 = time.time()
        processed_now = 0
        next_report_at = ((already_done // progress_every) + 1) * progress_every if progress_every > 0 else 0

        try:
            for batch in iter_pending_arrow_duel_rows(
                src_conn, out_conn, layout.row_encoding, batch_size=batch_size, limit=limit
            ):
                q_mat = scorer.score_batch(batch, layout.row_encoding, elos)
                orig_arr = np.fromiter((r.rating for r in batch), dtype=np.int32, count=len(batch))
                ad_full = solve_prompted_arrow_duel_batch(
                    orig_arr,
                    elos,
                    q_mat,
                    target_prob=DEFAULT_TARGET_PROB,
                    check_rate=DEFAULT_FULL_CHECK_RATE,
                )
                ad_sprint = solve_prompted_arrow_duel_batch(
                    orig_arr,
                    elos,
                    q_mat,
                    target_prob=DEFAULT_TARGET_PROB,
                    check_rate=DEFAULT_SPRINT_CHECK_RATE,
                )
                q_bp = np.clip(np.rint(q_mat * 10000.0), 0, 10000).astype(np.int32)

                insert_rows = [
                    (
                        batch[i].puzzle_id,
                        int(orig_arr[i]),
                        int(ad_full[i]),
                        int(ad_sprint[i]),
                        int(q_bp[i, 0]),
                        int(q_bp[i, 1]),
                        int(q_bp[i, 2]),
                        int(q_bp[i, 3]),
                        int(q_bp[i, 4]),
                    )
                    for i in range(len(batch))
                ]
                with out_conn:
                    out_conn.executemany(
                        """
                        INSERT OR REPLACE INTO arrow_duel_ratings (
                          puzzle_id, original_rating, arrow_duel_rating, arrow_duel_rating_sprint,
                          q_700_bp, q_1100_bp, q_1500_bp, q_1900_bp, q_2300_bp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        insert_rows,
                    )
                processed_now += len(batch)
                done_total = already_done + processed_now
                if progress_every > 0 and done_total >= next_report_at:
                    dt = time.time() - t0
                    rate = processed_now / max(1e-6, dt)
                    eta = (effective_total - done_total) / max(1e-6, rate)
                    print(
                        f"[progress] {done_total:,}/{effective_total:,} "
                        f"({done_total / max(1, effective_total) * 100:.1f}%) | "
                        f"{rate:,.0f} puzzles/s | elapsed={dt:.1f}s | eta={eta:.0f}s",
                        flush=True,
                    )
                    while next_report_at <= done_total:
                        next_report_at += progress_every

            elapsed = time.time() - t0
            out_conn.execute(
                "INSERT OR REPLACE INTO analysis_metadata (key, value) VALUES ('completedAt', ?)",
                (datetime.now(timezone.utc).isoformat(),),
            )
            out_conn.commit()
            report = build_report(
                out_conn,
                identity,
                total_pack_puzzles=layout.puzzle_count,
                total_eligible=total_eligible,
                elapsed_seconds=elapsed,
            )
            if report_path is not None:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            return report
        finally:
            out_conn.close()
    finally:
        src_conn.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Maia-backed Arrow Duel ratings for a Chessticize Core Pack SQLite DB."
    )
    parser.add_argument("input_pack", type=Path, help="Path to bundled-core-pack.sqlite")
    parser.add_argument("output_db", type=Path, help="Path to output sidecar SQLite database")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path")
    parser.add_argument("--model", default="5m", choices=["5m", "23m", "79m"], help="Maia-3 model alias")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke runs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    scorer = Maia3PairwiseScorer(model_alias=args.model, device=args.device)
    report = run_rating_job(
        input_path=args.input_pack,
        output_path=args.output_db,
        scorer=scorer,
        report_path=args.report,
        batch_size=args.batch_size,
        limit=args.limit,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
