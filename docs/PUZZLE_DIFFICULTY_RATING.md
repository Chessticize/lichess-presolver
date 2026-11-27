# Puzzle Difficulty Rating Methodology (Standard & Arrow Duel)

This document specifies Chessticize's offline puzzle difficulty rating system for both:
1. **Arrow Duel Core Pack rating (`rate_arrow_duel_puzzles.py`)** — recalculating Arrow Duel difficulty across all 1,392,969 Arrow Duel-eligible positions in the Core Pack.
2. **Personal-game puzzle difficulty estimation (`train_puzzle_difficulty_model.py`)** — predicting Lichess-aligned Standard and Arrow Duel ratings offline on-device so personal Game Review queues can filter out low-ROI puzzles far above a player's current rating.

## Why Separate Rating Models Are Needed

- **Personal game puzzles** extracted locally from a user's games have no crowd-calibrated Lichess rating. Assigning every extracted puzzle the player's game rating (or a flat 1500) mixes trivial 750-Elo mate-in-1 oversights with 1900+ engine-only quiet moves.
- **Arrow Duel vs. Lichess Standard Rating**: Lichess `Rating` measures how difficult it is to execute the tactical refutation *after* the blunder (`solutionMoves[0]`) has been played. Arrow Duel instead shows two highlighted arrows (`stockfishBestMove` vs. `solutionMoves[0]`) *before* the blunder is played.
- Structural Stockfish metrics alone (such as `top_move_unique_plies` in [ARROW_DUEL_ROUTE_METRICS.md](ARROW_DUEL_ROUTE_METRICS.md)) do not capture human move temptation or move blindness. Combining **human-policy models (Maia)** with **shallow-to-deep Stockfish search features** directly models human intuition and calculation thresholds.

---

## 1. Arrow Duel Rating Model (`rate_arrow_duel_puzzles.py`)

### First-Principles Prompted-Comparison Formula

When a player of rating $E$ is presented with two arrows ($m_+ = \text{stockfishBestMove}$ and $m_- = \text{solutionMoves}[0]$):

1. **Refutation check**: Because the two arrows focus attention on comparing $m_+$ and $m_-$, the player spots the tactical refutation of $m_-$ with probability:
   $$p_{\text{refute}}(E) = \alpha \cdot \sigma\!\left(\frac{E - R_{\text{refute}}}{173.7178}\right)$$
   where $R_{\text{refute}}$ is the Standard puzzle rating (Lichess `rating` for Core Pack puzzles, or predicted `standardRating` for user-game puzzles), $173.7178 = 400 / \ln(10)$ is the Glicko logit scale, and $\alpha \in (0, 1]$ is the check rate (`1.0` under full calculation; `0.75` under Sprint time pressure).
2. **Intuition fallback**: When the player cannot fully refute $m_-$, they choose between the two arrows according to human pairwise move preference from Maia-3 across the Elo grid $E \in \{700, 1100, 1500, 1900, 2300\}$:
   $$q(E) = \frac{P_E(m_+)}{P_E(m_+) + P_E(m_-)} = \sigma\big(z_E(m_+) - z_E(m_-)\big)$$
   Because the softmax normalizer cancels exactly in the pairwise ratio, `q(E)` is numerically identical (verified within `< 2e-6`) to full legal-masked softmax while avoiding legal move generation overhead.
3. **Total solve probability and 3PL-IRT anchor**:
   $$P_{\text{AD}}(E) = p_{\text{refute}}(E) + \big(1 - p_{\text{refute}}(E)\big) \cdot q(E)$$
   Solving $P_{\text{AD}}(E^*) = 0.75$ (clamped to `[600, 2200]` for Core Pack compatibility) aligns directly with the 3PL-IRT midpoint for a two-choice task ($c = 0.5$):
   - When human intuition is neutral ($q = 0.5$) and $\alpha = 1.0$, $P_{\text{AD}}(R_{\text{refute}}) = 0.5 + 0.5 \times 0.5 = 0.75$, so **$E^* = R_{\text{refute}}$ exactly**.
   - When the good move is intuitively obvious ($q > 0.75$), $E^*$ drops cleanly toward `600`, removing trivial giveaways from high rating buckets.
   - When the blunder is a deceptive human trap ($q \to 0$), $E^*$ rises above $R_{\text{refute}}$ (up to `+191 Elo` at $\alpha = 1.0$, or further under Sprint time pressure $\alpha = 0.75$).

### Sidecar Storage Contract

`rate_arrow_duel_puzzles.py` writes a self-describing SQLite sidecar (`arrow_duel_ratings`) keyed by `puzzle_id` containing:
- `original_rating`
- `arrow_duel_rating` ($\alpha = 1.0$, clamped to `[600, 2200]`)
- `arrow_duel_rating_sprint` ($\alpha = 0.75$, clamped to `[600, 2200]`)
- `q_700_bp`, `q_1100_bp`, `q_1500_bp`, `q_1900_bp`, `q_2300_bp` (pairwise preference in basis points `0..10000`)

Storing the raw 5-Elo pairwise preferences allows future recalibration of $\alpha$ or opponent-reply blending across all 1.39M puzzles in seconds without re-running Maia inference.

### Usage

```sh
python3 rate_arrow_duel_puzzles.py \
  ../chessticize-mobile/fixtures/puzzles/bundled-core-pack.sqlite \
  arrow-duel-ratings-maia3-5m.sqlite \
  --report arrow-duel-ratings-maia3-5m.report.json \
  --model 5m \
  --device auto \
  --batch-size 1024
```

Interrupted runs can resume with `--resume` only when input SHA-256, model checkpoint SHA-256, and formula identity match exactly.

---

## 2. Offline Standard Difficulty Model (`train_puzzle_difficulty_model.py`)

Trained on **51,200 stratified Core Pack puzzles** (3,200 per 100-Elo bucket across 600–2200, `RatingDeviation <= 90`, 80/20 train/validation split) using compact `GradientBoostingRegressor` ensembles (`n_estimators=120, max_depth=4`) exported as zero-dependency pure JSON trees for TypeScript inference in `chessticize-mobile`.

### Feature Families

1. **Structural features (14 features, 0 ms):** solution plies, solver moves, piece count, legal move count, queens on board, endgame flag, first-move capture/check/mate-in-1/quiet/promotion/piece value/captured value/sacrifice.
2. **Stockfish shallow-to-deep features (9 features):** smallest depth in `{1, 2, 4, 8}` that finds `solutionMoves[1]`, solver-POV evals at depth `1, 2, 4, 8`, eval jumps (`d8 - d1`, `d8 - d2`, `d8 - d4`), and MultiPV-2 gap at depth `8`.
3. **Maia-2 Mobile-Lite features (18 features, MIT license):** only **2 Elo queries (`1100` and `1700`)** per solver position (first move probability/log-prob/rank/log-ratio vs. top human move/entropy, full line log-prob/min-prob/max-rank, and 1700-vs-1100 skill slope).

### Held-Out Validation Results (`n = 10,240`, 600–2200 Elo)

| Model Artifact | Size (JSON) | MAE (Elo) | RMSE (Elo) | Pearson `r` | `AUC (>=1800)` | Drop `>=1900` (@1750) | False-drop `<=1550` (@1750) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sf_heuristic_v1` (fallback, no Maia) | 79.9 KB | 269.70 | 332.88 | 0.6889 | 0.8116 | 35.7% | 3.81% |
| `maia2_sf_v1` (MIT Maia-2, 2 Elos + SF) | 84.2 KB | **223.18** | **282.15** | **0.7891** | **0.8771** | **58.3%** | **3.76%** |

### Maia-2 INT8 ONNX Quantization Verification (On-Device Mobile Target)

Exported via `train_puzzle_difficulty_model.py --export-onnx` and verified on 1,000 stratified puzzles on CPU-only ONNX Runtime:

- **FP32 ONNX size:** 84.4 MB (`84,432,911` bytes)
- **INT8 quantized ONNX size:** **21.3 MB** (`21,323,255` bytes, 4.0× smaller)
- **INT8 SHA-256:** `sha256:3e93650b4db4ffa9a267a77346bd177c7493b83311d2e97ef00739636723d92c`
- **Quantization impact:** `maeDelta = -1.09 Elo` (`216.46` vs `217.55`), `aucDelta = +0.0020` (`0.8897` vs `0.8877`) — **zero accuracy degradation**.
- **CPU throughput:** 1,000 multi-move puzzles scored in `6.98 s` (~7 ms/puzzle on desktop ARM64 CPU; single-threaded 1-position PyTorch CPU latency is `3.6 ms` for Maia-2 and `3.1 ms` for Maia-3-5M).

---

## 3. Full Core Pack Arrow Duel Recalculation (`1,392,969` rows)

Executed on `core-pack-v5` (`sha256:4f8726cd64c8e490708f9c6b7b411dad3736d5936c0493d71fd42bbe4404a811`, `binary-v1`, 1,400,000 total rows, **1,392,969** Arrow Duel-eligible rows) with `UofTCSSLab/Maia3-5M` (`sha256:ba14208b2992d85502f5fb501934abf6aaaeb355e9f3fdf90e326911f562524f`) on Apple Silicon MPS in **23.1 minutes** (`1,386.09 s`).

### Aggregate Summary

- **Maia-3 intuition at 1500 Elo:**
  - Blunder looks more tempting than best move (`q < 0.35`): **46.12%**
  - Balanced / ambiguous (`0.35 <= q <= 0.65`): **24.16%**
  - Best move already looks obvious (`q > 0.65`): **29.73%** (overwhelmingly obvious `q > 0.85`: **14.30%**)
- **Full-calculation Arrow Duel rating (`check = 1.0`, `[600, 2200]`):**
  - Mean `1299.4`, std `512.7`, Pearson **`r = 0.5968`**, Spearman **`rho = 0.5746`**
  - **78.22%** remain within `±300 Elo` of original Lichess rating (median shift `+53 Elo`, MAD `157 Elo`)
  - **18.94%** drop by `>= 400 Elo` (removing intuitive giveaways from high rating buckets)
- **Sprint time-pressure Arrow Duel rating (`check = 0.75`, `[600, 2200]`):**
  - Mean `1468.1`, std `573.4`, Pearson **`r = 0.4145`**, Spearman **`rho = 0.4426`**
  - **18.64%** drop by `>= 400 Elo` and **24.44%** rise by `>= 400 Elo` (deceptive human traps)

### Per Original Lichess Bucket (`check = 1.0`)

| Original Lichess Bucket | Count | AD Mean | AD p10 | AD Median | AD p90 | Share Trivial (`<= 900`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `600–799` | 109,156 | 763.7 | 600 | 785 | 922 | 84.98% |
| `800–999` | 181,225 | 916.6 | 600 | 976 | 1122 | 35.96% |
| `1000–1199` | 186,338 | 1075.1 | 600 | 1173 | 1319 | 22.16% |
| `1200–1399` | 185,091 | 1221.1 | 600 | 1357 | 1512 | 19.37% |
| `1400–1599` | 186,272 | 1361.8 | 600 | 1547 | 1708 | 19.38% |
| `1600–1799` | 184,652 | 1495.0 | 600 | 1732 | 1906 | 19.94% |
| `1800–1999` | 181,730 | 1627.3 | 600 | 1915 | 2100 | 20.09% |
| `2000–2200` | 178,505 | 1729.8 | 600 | 2093 | 2200 | 20.33% |

Every 100-Elo bucket in the recalculated `arrow_duel_rating` distribution retains ample inventory (minimum `45,174` in `700–799`, `58,371` in `2000–2099`, `106,463` in `2100–2200`, and `265,005` in `600–699`).

The full sidecar SQLite (`arrow-duel-ratings-maia3-5m.sqlite.zst`), JSON report, and SHA-256 checksums are published under release tag `arrow-duel-ratings-maia3-5m-2026-11-27` in this repository.
