#!/usr/bin/env python3
"""Train compact pure-JSON puzzle difficulty models and verify Maia-2 INT8 ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import chess
import chess.engine
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

from classify_arrow_duel_routes import (
    arrow_duel_input_sql,
    decode_pack_move,
    decode_pack_position,
    validate_pack,
)

MAIA2_MOBILE_ELOS: tuple[int, int] = (1100, 1700)
SF_SHALLOW_DEPTHS: tuple[int, ...] = (1, 2, 4, 8)
PIECE_VALUES: dict[int, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

STRUCTURAL_FEATURE_NAMES: tuple[str, ...] = (
    "n_solution_moves",
    "n_solver_moves",
    "n_pieces",
    "n_legal_moves_solver",
    "queens_on_board",
    "is_endgame",
    "std_m1_is_capture",
    "std_m1_is_check",
    "std_m1_is_mate1",
    "std_m1_is_quiet",
    "std_m1_is_promo",
    "std_m1_piece_val",
    "std_m1_cap_val",
    "std_m1_is_sac",
)

SF_STD_FEATURE_NAMES: tuple[str, ...] = (
    "sf_std_found_depth",
    "sf_std_eval_d1",
    "sf_std_eval_d2",
    "sf_std_eval_d4",
    "sf_std_eval_d8",
    "sf_std_jump_d8_d1",
    "sf_std_jump_d8_d2",
    "sf_std_jump_d8_d4",
    "sf_std_multipv_gap_d8",
)

MAIA2_STD_FEATURE_NAMES: tuple[str, ...] = (
    "m2_m1_p_1100",
    "m2_m1_logp_1100",
    "m2_m1_rank_1100",
    "m2_m1_logratio_1100",
    "m2_line_logp_1100",
    "m2_line_minp_1100",
    "m2_line_maxrank_1100",
    "m2_m1_ent_1100",
    "m2_m1_p_1700",
    "m2_m1_logp_1700",
    "m2_m1_rank_1700",
    "m2_m1_logratio_1700",
    "m2_line_logp_1700",
    "m2_line_minp_1700",
    "m2_line_maxrank_1700",
    "m2_m1_ent_1700",
    "m2_m1_slope",
    "m2_line_logp_slope",
)


def decode_pack_move_line(blob: bytes) -> list[str]:
    if len(blob) % 2 != 0:
        raise ValueError(f"Invalid binary move line length {len(blob)}")
    return [decode_pack_move(blob[i : i + 2]) for i in range(0, len(blob), 2)]


def sample_stratified_puzzles(
    pack_path: Path,
    per_bucket: int = 2000,
    seed: str = "difficulty-train-v1",
) -> list[dict[str, object]]:
    conn = sqlite3.connect(f"file:{pack_path}?mode=ro", uri=True)
    try:
        layout = validate_pack(conn)
        if layout.row_encoding != "binary-v1":
            raise ValueError(f"Expected binary-v1 pack, got {layout.row_encoding}")

        puzzles: list[dict[str, object]] = []
        for lo in range(600, 2200, 100):
            hi = 2200 if lo == 2100 else lo + 99
            rows = conn.execute(
                """
                SELECT id, initial_fen, solution_moves, rating, rating_deviation, stockfish_bestmove
                FROM puzzles
                WHERE rating >= ? AND rating <= ? AND rating_deviation <= 90
                ORDER BY id
                """,
                (lo, hi),
            ).fetchall()
            # Deterministic hash order
            rows.sort(key=lambda r: hashlib.sha256(f"{seed}:{r[0]}".encode()).digest())
            for r in rows[:per_bucket]:
                puzzles.append(
                    {
                        "id": r[0],
                        "initialFen": decode_pack_position(r[1]),
                        "solutionMoves": decode_pack_move_line(r[2]),
                        "rating": int(r[3]),
                        "ratingDeviation": int(r[4]),
                        "stockfishBestMove": decode_pack_move(r[5]),
                    }
                )
        return puzzles
    finally:
        conn.close()


def clamp_cp(score: chess.engine.PovScore, pov: chess.Color) -> float:
    s = score.pov(pov)
    if s.is_mate():
        m = s.mate()
        return 1500.0 if (m is not None and m > 0) else -1500.0
    cp = s.score()
    return 0.0 if cp is None else float(max(-1500, min(1500, cp)))


def extract_structural_vector(p: dict[str, object]) -> list[float]:
    b0 = chess.Board(str(p["initialFen"]))
    moves = list(p["solutionMoves"])  # type: ignore[arg-type]
    b1 = b0.copy()
    b1.push_uci(moves[0])
    first_m = chess.Move.from_uci(moves[1]) if len(moves) > 1 else None

    is_cap = is_chk = is_mate1 = is_quiet = is_promo = is_sac = 0.0
    mv_val = cap_val = 0.0
    if first_m is not None:
        cap = b1.is_capture(first_m)
        chk = b1.gives_check(first_m)
        b_after = b1.copy()
        b_after.push(first_m)
        mate1 = b_after.is_checkmate()
        moved_piece = b1.piece_at(first_m.from_square)
        cap_piece = b1.piece_at(first_m.to_square)
        mv_v = PIECE_VALUES.get(moved_piece.piece_type, 0) if moved_piece else 0
        cap_v = PIECE_VALUES.get(cap_piece.piece_type, 0) if cap_piece else (1 if b1.is_en_passant(first_m) else 0)
        sac = mv_v > cap_v and b_after.is_attacked_by(b_after.turn, first_m.to_square)
        is_cap = float(cap)
        is_chk = float(chk)
        is_mate1 = float(mate1)
        is_quiet = float((not cap) and (not chk))
        is_promo = float(first_m.promotion is not None)
        mv_val = float(mv_v)
        cap_val = float(cap_v)
        is_sac = float(sac)

    n_pieces = len(b1.piece_map())
    queens = len(b1.pieces(chess.QUEEN, chess.WHITE)) + len(b1.pieces(chess.QUEEN, chess.BLACK))
    return [
        float(len(moves)),
        float(len(moves) // 2),
        float(n_pieces),
        float(b1.legal_moves.count()),
        float(queens),
        1.0 if n_pieces <= 12 else 0.0,
        is_cap,
        is_chk,
        is_mate1,
        is_quiet,
        is_promo,
        mv_val,
        cap_val,
        is_sac,
    ]


def _sf_worker(args: tuple[str, list[tuple[int, str, list[str]]]]) -> list[tuple[int, list[float]]]:
    sf_path, items = args
    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    engine.configure({"Threads": 1, "Hash": 16})
    out: list[tuple[int, list[float]]] = []
    try:
        for idx, fen0, moves in items:
            b = chess.Board(fen0)
            b.push_uci(moves[0])
            solver_color = b.turn
            first_sol = moves[1] if len(moves) > 1 else None
            found_depth = 12.0
            evals: dict[int, float] = {}
            for d in SF_SHALLOW_DEPTHS:
                info = engine.analyse(b, chess.engine.Limit(depth=d))
                evals[d] = clamp_cp(info["score"], solver_color)
                pv = info.get("pv")
                bm = pv[0].uci() if pv else None
                if first_sol and bm == first_sol and found_depth == 12.0:
                    found_depth = float(d)
            info_m2 = engine.analyse(b, chess.engine.Limit(depth=8), multipv=2)
            if isinstance(info_m2, list) and len(info_m2) >= 2:
                gap_d8 = clamp_cp(info_m2[0]["score"], solver_color) - clamp_cp(info_m2[1]["score"], solver_color)
            else:
                gap_d8 = 1500.0
            out.append(
                (
                    idx,
                    [
                        found_depth,
                        evals[1],
                        evals[2],
                        evals[4],
                        evals[8],
                        evals[8] - evals[1],
                        evals[8] - evals[2],
                        evals[8] - evals[4],
                        gap_d8,
                    ],
                )
            )
    finally:
        engine.quit()
    return out


def extract_sf_features_parallel(
    puzzles: Sequence[dict[str, object]],
    stockfish_path: str,
    workers: int = 8,
) -> np.ndarray:
    items = [(i, str(p["initialFen"]), list(p["solutionMoves"])) for i, p in enumerate(puzzles)]  # type: ignore[arg-type]
    chunks = [(stockfish_path, items[w::workers]) for w in range(workers)]
    res = np.zeros((len(puzzles), len(SF_STD_FEATURE_NAMES)), dtype=np.float32)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for batch in ex.map(_sf_worker, chunks):
            for idx, vec in batch:
                res[idx] = vec
    return res


def extract_maia2_features(
    puzzles: Sequence[dict[str, object]],
    *,
    device: str = "auto",
    onnx_path: Path | None = None,
    batch_size: int = 512,
) -> np.ndarray:
    """Extract 2-Elo Mobile-Lite Maia-2 features via PyTorch or ONNX Runtime."""
    import torch
    from maia2 import inference as m2_inf
    from maia2 import model as m2_model

    all_moves_dict, elo_dict, _ = m2_inf.prepare()
    elo_cats = [m2_inf.map_to_category(e, elo_dict) for e in MAIA2_MOBILE_ELOS]

    # Collect solver positions (up to 4 solver moves per puzzle)
    fens: list[str] = []
    target_idxs: list[int] = []
    slices: list[tuple[int, int]] = []
    for p in puzzles:
        b = chess.Board(str(p["initialFen"]))
        moves = list(p["solutionMoves"])  # type: ignore[arg-type]
        b.push_uci(moves[0])
        s0 = len(fens)
        for k in range(1, min(len(moves), 9), 2):
            fens.append(b.fen())
            u = m2_inf.mirror_move(moves[k]) if b.turn == chess.BLACK else moves[k]
            target_idxs.append(all_moves_dict.get(u, -1))
            b.push_uci(moves[k])
            if k + 1 < len(moves):
                b.push_uci(moves[k + 1])
        slices.append((s0, len(fens)))

    n_pos = len(fens)
    # Per-position stats for each of the 2 Elos: (p, rank, top1_p, entropy)
    pos_stats = np.zeros((n_pos, 2, 4), dtype=np.float32)

    if onnx_path is not None:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        for start in range(0, n_pos, batch_size):
            end = min(n_pos, start + batch_size)
            bt_list = []
            lm_list = []
            for f in fens[start:end]:
                bt, _, _, lm = m2_inf.preprocessing(f, 1500, 1500, elo_dict, all_moves_dict)
                bt_list.append(bt.numpy())
                lm_list.append(lm.numpy())
            bt_np = np.stack(bt_list, axis=0).astype(np.float32)
            lm_np = np.stack(lm_list, axis=0)
            for e_i, elo_cat in enumerate(elo_cats):
                el_np = np.full((end - start,), elo_cat, dtype=np.int64)
                logits = sess.run(None, {"boards": bt_np, "elo_self": el_np, "elo_oppo": el_np})[0]
                logits = np.where(lm_np == 0, -1e9, logits)
                logits -= logits.max(axis=1, keepdims=True)
                exp = np.exp(logits)
                probs = exp / np.maximum(1e-12, exp.sum(axis=1, keepdims=True))
                for j in range(end - start):
                    pr = probs[j]
                    midx = target_idxs[start + j]
                    p_m = float(pr[midx]) if midx >= 0 else 1e-6
                    r_m = float((pr > p_m).sum() + 1)
                    t1 = float(pr.max())
                    nz = pr[pr > 1e-9]
                    ent = float(-(nz * np.log(nz)).sum())
                    pos_stats[start + j, e_i] = (p_m, r_m, t1, ent)
    else:
        resolved = "mps" if (device == "auto" and torch.backends.mps.is_available()) else ("cpu" if device == "auto" else device)
        m2 = m2_model.from_pretrained(type="rapid", device=resolved)
        m2.eval()
        with torch.no_grad():
            for start in range(0, n_pos, batch_size):
                end = min(n_pos, start + batch_size)
                bt_list = []
                lm_list = []
                for f in fens[start:end]:
                    bt, _, _, lm = m2_inf.preprocessing(f, 1500, 1500, elo_dict, all_moves_dict)
                    bt_list.append(bt)
                    lm_list.append(lm)
                bt_t = torch.stack(bt_list, dim=0).to(resolved)
                lm_t = torch.stack(lm_list, dim=0).to(resolved)
                for e_i, elo_cat in enumerate(elo_cats):
                    el_t = torch.full((end - start,), elo_cat, dtype=torch.long, device=resolved)
                    logits, _, _ = m2(bt_t, el_t, el_t)
                    logits = logits.float().masked_fill(lm_t == 0, float("-inf"))
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    for j in range(end - start):
                        pr = probs[j]
                        midx = target_idxs[start + j]
                        p_m = float(pr[midx]) if midx >= 0 else 1e-6
                        r_m = float((pr > p_m).sum() + 1)
                        t1 = float(pr.max())
                        nz = pr[pr > 1e-9]
                        ent = float(-(nz * np.log(nz)).sum())
                        pos_stats[start + j, e_i] = (p_m, r_m, t1, ent)

    eps = 1e-6
    out = np.zeros((len(puzzles), len(MAIA2_STD_FEATURE_NAMES)), dtype=np.float32)
    for i, (s0, s1) in enumerate(slices):
        row_vals: list[float] = []
        p1_list = []
        lp_list = []
        for e_i in range(2):
            sub = pos_stats[s0:s1, e_i]
            if len(sub) == 0:
                ps, rs, t1s, ents = [0.5], [1.0], [0.5], [1.0]
            else:
                ps = sub[:, 0].tolist()
                rs = sub[:, 1].tolist()
                t1s = sub[:, 2].tolist()
                ents = sub[:, 3].tolist()
            line_lp = sum(math.log(x + eps) for x in ps)
            row_vals.extend(
                [
                    ps[0],
                    math.log(ps[0] + eps),
                    rs[0],
                    math.log((ps[0] + eps) / (t1s[0] + eps)),
                    line_lp,
                    min(ps),
                    max(rs),
                    ents[0],
                ]
            )
            p1_list.append(ps[0])
            lp_list.append(line_lp)
        row_vals.append(p1_list[1] - p1_list[0])
        row_vals.append(lp_list[1] - lp_list[0])
        out[i] = row_vals
    return out


def export_gbr_to_json(
    reg: GradientBoostingRegressor,
    feature_names: Sequence[str],
    *,
    model_id: str,
    metrics: dict[str, float],
    calibration: dict[str, float],
) -> dict[str, object]:
    """Export sklearn GradientBoostingRegressor to compact, zero-dependency pure JSON trees."""
    trees = []
    for est_arr in reg.estimators_:
        t = est_arr[0].tree_
        trees.append(
            {
                "f": [int(x) for x in t.feature],
                "t": [round(float(x), 5) for x in t.threshold],
                "l": [int(x) for x in t.children_left],
                "r": [int(x) for x in t.children_right],
                "v": [round(float(x[0][0]), 4) for x in t.value],
            }
        )
    init_pred = float(reg.init_.constant_[0][0])
    return {
        "schemaVersion": 1,
        "modelId": model_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "featureNames": list(feature_names),
        "learningRate": float(reg.learning_rate),
        "initValue": round(init_pred, 4),
        "minRating": 600,
        "maxRating": 2200,
        "linearCalibration": calibration,
        "metrics": metrics,
        "trees": trees,
    }


def predict_exported_gbr(artifact: dict[str, object], X: np.ndarray) -> np.ndarray:
    lr = float(artifact["learningRate"])  # type: ignore[arg-type]
    init_v = float(artifact["initValue"])  # type: ignore[arg-type]
    trees = artifact["trees"]  # type: ignore[assignment]
    preds = np.full((len(X),), init_v, dtype=np.float64)
    for i in range(len(X)):
        row = X[i]
        acc = init_v
        for tree in trees:  # type: ignore[union-attr]
            f = tree["f"]
            thr = tree["t"]
            left = tree["l"]
            right = tree["r"]
            val = tree["v"]
            node = 0
            while left[node] != -1:
                if row[f[node]] <= thr[node]:
                    node = left[node]
                else:
                    node = right[node]
            acc += lr * val[node]
        preds[i] = acc
    cal = artifact["linearCalibration"]  # type: ignore[assignment]
    slope = float(cal["slope"])  # type: ignore[index]
    intercept = float(cal["intercept"])  # type: ignore[index]
    return np.clip(preds * slope + intercept, float(artifact["minRating"]), float(artifact["maxRating"]))  # type: ignore[arg-type]


def export_maia2_int8_onnx(out_dir: Path) -> tuple[Path, Path]:
    """Export Maia-2 Rapid to FP32 ONNX and dynamic-INT8 quantized ONNX."""
    import torch
    from maia2 import model as m2_model
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = out_dir / "maia2-rapid-fp32.onnx"
    int8_path = out_dir / "maia2-rapid-int8.onnx"

    m2 = m2_model.from_pretrained(type="rapid", device="cpu")
    m2.eval()

    class PolicyOnlyWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, boards, elo_self, elo_oppo):
            logits, _, _ = self.inner(boards, elo_self, elo_oppo)
            return logits

    wrapper = PolicyOnlyWrapper(m2)
    dummy_boards = torch.zeros((2, 18, 8, 8), dtype=torch.float32)
    dummy_e1 = torch.tensor([5, 5], dtype=torch.long)
    dummy_e2 = torch.tensor([5, 5], dtype=torch.long)

    torch.onnx.export(
        wrapper,
        (dummy_boards, dummy_e1, dummy_e2),
        str(fp32_path),
        input_names=["boards", "elo_self", "elo_oppo"],
        output_names=["logits"],
        dynamic_axes={
            "boards": {0: "batch"},
            "elo_self": {0: "batch"},
            "elo_oppo": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QUInt8,
    )
    return fp32_path, int8_path


def fit_and_export(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    model_id: str,
    *,
    n_estimators: int = 120,
    max_depth: int = 4,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=0.07,
        max_depth=max_depth,
        min_samples_leaf=30,
        subsample=0.85,
        random_state=42,
    )
    reg.fit(X_tr, y_tr)
    raw_val = reg.predict(X_val)

    # Mild variance-preserving linear calibration only if it doesn't hurt MAE; keep identity by default
    cal = {"slope": 1.0, "intercept": 0.0}
    preds = np.clip(raw_val, 600.0, 2200.0)
    mae = float(mean_absolute_error(y_val, preds))
    rmse = float(root_mean_squared_error(y_val, preds))
    r_val = float(np.corrcoef(y_val, preds)[0, 1])
    auc_1800 = float(roc_auc_score((y_val >= 1800).astype(int), preds))
    drop_1900 = float((preds[y_val >= 1900] >= 1750).mean())
    false_drop_1550 = float((preds[y_val <= 1550] >= 1750).mean())

    metrics = {
        "trainSize": int(len(X_tr)),
        "valSize": int(len(X_val)),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "pearsonR": round(r_val, 4),
        "aucGe1800": round(auc_1800, 4),
        "dropGe1900At1750": round(drop_1900, 4),
        "falseDropLe1550At1750": round(false_drop_1550, 4),
    }

    # Refit on full sample for exported artifact
    reg_full = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=0.07,
        max_depth=max_depth,
        min_samples_leaf=30,
        subsample=0.85,
        random_state=42,
    )
    reg_full.fit(X, y)
    artifact = export_gbr_to_json(reg_full, feature_names, model_id=model_id, metrics=metrics, calibration=cal)
    return artifact, X_val, y_val


def main() -> int:
    parser = argparse.ArgumentParser(description="Train compact difficulty models & verify Maia-2 INT8 ONNX.")
    parser.add_argument("pack_path", type=Path, help="Path to bundled-core-pack.sqlite")
    parser.add_argument("out_dir", type=Path, help="Output directory for JSON models and report")
    parser.add_argument("--per-bucket", type=int, default=2000, help="Puzzles per 100-Elo bucket (x16 buckets)")
    parser.add_argument("--stockfish-path", default="stockfish")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--export-onnx", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    puzzles = sample_stratified_puzzles(args.pack_path, per_bucket=args.per_bucket)
    y = np.array([int(p["rating"]) for p in puzzles], dtype=np.float32)
    print(f"Sampled {len(puzzles):,} stratified puzzles in {time.time() - t0:.1f}s", flush=True)

    t1 = time.time()
    X_struct = np.array([extract_structural_vector(p) for p in puzzles], dtype=np.float32)
    print(f"Structural features done in {time.time() - t1:.1f}s", flush=True)

    t2 = time.time()
    X_sf = extract_sf_features_parallel(puzzles, args.stockfish_path, workers=args.workers)
    print(f"Stockfish shallow-deep features done in {time.time() - t2:.1f}s", flush=True)

    t3 = time.time()
    X_m2 = extract_maia2_features(puzzles, device=args.device)
    print(f"Maia-2 Mobile-Lite (2-Elo) features done in {time.time() - t3:.1f}s", flush=True)

    # 1) SF-only fallback model
    sf_names = list(SF_STD_FEATURE_NAMES) + list(STRUCTURAL_FEATURE_NAMES)
    X_sf_all = np.concatenate([X_sf, X_struct], axis=1)
    art_sf, _, _ = fit_and_export(X_sf_all, y, sf_names, "sf_heuristic_v1")
    sf_json_path = args.out_dir / "difficulty-model-sf-heuristic-v1.json"
    sf_json_path.write_text(json.dumps(art_sf, separators=(",", ":")), encoding="utf-8")
    print(f"Exported {sf_json_path.name} ({sf_json_path.stat().st_size / 1024:.1f} KB) metrics={art_sf['metrics']}", flush=True)

    # 2) Maia-2 + SF model
    m2_names = list(MAIA2_STD_FEATURE_NAMES) + list(SF_STD_FEATURE_NAMES) + list(STRUCTURAL_FEATURE_NAMES)
    X_m2_all = np.concatenate([X_m2, X_sf, X_struct], axis=1)
    art_m2, X_m2_val, y_m2_val = fit_and_export(X_m2_all, y, m2_names, "maia2_sf_v1")
    m2_json_path = args.out_dir / "difficulty-model-maia2-sf-v1.json"
    m2_json_path.write_text(json.dumps(art_m2, separators=(",", ":")), encoding="utf-8")
    print(f"Exported {m2_json_path.name} ({m2_json_path.stat().st_size / 1024:.1f} KB) metrics={art_m2['metrics']}", flush=True)

    report: dict[str, object] = {
        "sampleCount": len(puzzles),
        "perBucket": args.per_bucket,
        "sfHeuristicV1": {
            "bytes": sf_json_path.stat().st_size,
            "metrics": art_sf["metrics"],
        },
        "maia2SfV1": {
            "bytes": m2_json_path.stat().st_size,
            "metrics": art_m2["metrics"],
        },
    }

    # 3) Optional Maia-2 INT8 ONNX export & quantization verification on 1,000 validation puzzles
    if args.export_onnx:
        t4 = time.time()
        fp32_path, int8_path = export_maia2_int8_onnx(args.out_dir / "onnx")
        print(
            f"Exported ONNX FP32={fp32_path.stat().st_size / 1e6:.1f} MB, "
            f"INT8={int8_path.stat().st_size / 1e6:.1f} MB in {time.time() - t4:.1f}s",
            flush=True,
        )
        # Verify INT8 ONNX feature parity & end-to-end MAE/AUC on 1,000 puzzles
        sub_puzzles = puzzles[:: max(1, len(puzzles) // 1000)][:1000]
        sub_idx = list(range(0, len(puzzles), max(1, len(puzzles) // 1000)))[:1000]
        t5 = time.time()
        X_m2_int8 = extract_maia2_features(sub_puzzles, onnx_path=int8_path, batch_size=256)
        dt_int8 = time.time() - t5
        X_fp32_sub = X_m2_all[sub_idx]
        X_int8_sub = np.concatenate([X_m2_int8, X_sf[sub_idx], X_struct[sub_idx]], axis=1)
        y_sub = y[sub_idx]
        p_fp32 = predict_exported_gbr(art_m2, X_fp32_sub)
        p_int8 = predict_exported_gbr(art_m2, X_int8_sub)
        mae_fp32 = float(mean_absolute_error(y_sub, p_fp32))
        mae_int8 = float(mean_absolute_error(y_sub, p_int8))
        auc_fp32 = float(roc_auc_score((y_sub >= 1800).astype(int), p_fp32))
        auc_int8 = float(roc_auc_score((y_sub >= 1800).astype(int), p_int8))
        max_pred_diff = float(np.abs(p_fp32 - p_int8).max())
        mean_pred_diff = float(np.abs(p_fp32 - p_int8).mean())
        onnx_report = {
            "fp32Bytes": fp32_path.stat().st_size,
            "int8Bytes": int8_path.stat().st_size,
            "int8Sha256": f"sha256:{hashlib.sha256(int8_path.read_bytes()).hexdigest()}",
            "verifyPuzzles": len(sub_puzzles),
            "int8CpuElapsedSec": round(dt_int8, 2),
            "fp32Mae": round(mae_fp32, 2),
            "int8Mae": round(mae_int8, 2),
            "maeDelta": round(mae_int8 - mae_fp32, 2),
            "fp32AucGe1800": round(auc_fp32, 4),
            "int8AucGe1800": round(auc_int8, 4),
            "aucDelta": round(auc_int8 - auc_fp32, 4),
            "meanPredDiffElo": round(mean_pred_diff, 2),
            "maxPredDiffElo": round(max_pred_diff, 2),
        }
        report["maia2Int8OnnxVerification"] = onnx_report
        print(f"ONNX INT8 verification: {json.dumps(onnx_report, indent=2)}", flush=True)

    report_path = args.out_dir / "training-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved training report to {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
