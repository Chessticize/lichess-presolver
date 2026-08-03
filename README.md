# Lichess Puzzle Presolver

Lichess Puzzle Presolver adds reproducible Stockfish evaluations to a snapshot
of the [Lichess puzzle database](https://database.lichess.org/#puzzles). It can
process one CSV, split a large snapshot, or run several split files in parallel.

The repository also distributes a presolved July 2025 snapshot as compressed
[GitHub Release assets](https://github.com/Chessticize/lichess-presolver/releases).
Only the later **depth-20** corpus is distributed. A complete comparison found
that the older `presolved-depth16` corpus was not the better-quality set; see
[Dataset provenance and quality](docs/DATASET.md).

## Download the presolved data

Download all eight independently decompressible assets from the latest dataset
release:

```sh
gh release download dataset-2025-07-depth20 \
  --repo Chessticize/lichess-presolver
shasum -a 256 -c SHA256SUMS
unzstd lichess-puzzles-presolved-depth20-2025-07-split-*.csv.zst
```

The release contains 4,824,507 puzzle rows split round-robin across eight CSV
files. Verify downloads with the release's `SHA256SUMS` file. The original
Lichess columns are preserved and three fields are appended:

| Column | Meaning |
| --- | --- |
| `stockfish_eval` | Evaluation of the source FEN, in centipawns from White's point of view |
| `stockfish_bestmove` | Stockfish's best move in the source FEN, in UCI notation |
| `stockfish_eval_after_first_move` | Evaluation after applying the first move in `Moves`, from White's point of view |

Forced mates are represented as `+10000` or `-10000`; the distance to mate is
not retained.

## Run the presolver

Requirements:

- Python 3.10 or newer
- Stockfish available as `stockfish`, or an explicit `--stockfish-path`

No third-party Python package is required.

```sh
python3 presolve_lichess_puzzles.py \
  puzzles/example.csv \
  example-presolved.csv \
  --depth 20
```

The command writes a metadata sidecar containing the input and output SHA-256
digests, engine identity, depth, thread count, hash size, timestamps, and row
count. Existing outputs are protected unless `--overwrite` is passed.

For a full snapshot:

```sh
python3 split_csv_util.py lichess_db_puzzle.csv 8 \
  --output-prefix puzzles/split_
python3 solve_parallel.py \
  --input-dir puzzles \
  --output-dir presolved-depth20 \
  --depth 20 \
  --jobs 8
```

## Measure Arrow Duel top-move route depth

`classify_arrow_duel_routes.py` measures how constrained the stored top-move
route is. It analyzes only the Arrow Duel-compatible subset of a generated
Core Pack; Core Pack quality, legality, and evaluation-gap checks remain owned
by the pack builder, and Standard-only promotion candidates never reach
Stockfish. Both legacy text rows and the versioned binary-v1 Core Pack encoding
are supported; unknown pack formats fail closed.

```sh
python3 classify_arrow_duel_routes.py \
  ../chessticize-mobile/fixtures/puzzles/bundled-core-pack.sqlite \
  arrow-duel-routes.sqlite \
  --report arrow-duel-routes.report.json \
  --compact-output arrow-duel-routes.compact.sqlite \
  --depth 10 \
  --multipv 2 \
  --jobs 10
```

The analysis stores `top_move_unique_plies` for each Arrow Duel input. It
follows the Core Pack's stored top move, takes the opponent's best replies, and
counts plies until the chooser first has multiple good continuations or the
line terminates. Root alternatives are deliberately ignored because Arrow Duel
shows only the stored top move and the puzzle's blunder. A unique mate in one
has depth one. `max_unique_plies + 1` means no branch or terminal was found
inside the analysis window. `unique_followup_moves` is derived as
`floor((top_move_unique_plies - 1) / 2)` and excludes the displayed top move.
With the default ten-ply window, the last bucket is `4+`: four forced
puzzle-side follow-ups are established, but the route is censored before the
next choice.

Only the best and second-best continuations are searched. If the second-best
line is outside the accepted score or mate-distance threshold, every
lower-ranked move is also outside it; if it is inside, uniqueness is already
disproved.

These are engine- and depth-qualified route metrics, not a universal difficulty
class. The audit SQLite sidecar keeps metric definitions in
`arrow_duel_metric_definitions` and row values in
`arrow_duel_route_metrics`. The JSON report records count distributions,
unique follow-up counts, rating buckets, and deterministic sample IDs. The
optional compact SQLite output contains no puzzle-ID strings: it binds
to the exact input hash and packs two Core Pack `puzzles.rowid` values per byte
in 4,096-row BLOB chunks. Each four-bit value is the top-move route depth; zero
means that row is outside the Arrow Duel input set. The metric definition
remains normalized in one small definition table.

Each puzzle starts a fresh UCI game before analysis. This prevents a worker's
transposition-table history from changing another puzzle's result. Interrupted
runs can continue with `--resume` only when the input hash, engine identity,
depth, job layout, and metric thresholds still match. The input SHA-256 is
checked again after analysis and before compact export, so a Core Pack replaced
during a run cannot produce a completed report. The original presolver
continues to read and write CSV; only this Core Pack analyzer uses SQLite.

See [Usage guide](docs/USAGE.md) for installation, resource sizing, CSV
semantics, verification, and troubleshooting. The
[Arrow Duel route metric note](docs/ARROW_DUEL_ROUTE_METRICS.md) records the
root-exclusion contract and the depth 10/16/20 stability and speed benchmark.

## Project documents

- [Usage guide](docs/USAGE.md)
- [Dataset provenance and quality](docs/DATASET.md)
- [Contributing](CONTRIBUTING.md)

## Licensing

Repository software is licensed under the [MIT License](LICENSE). Lichess
database exports and the derived release data are distributed under
[CC0 1.0](DATA_LICENSE.md). Lichess attribution is included as a courtesy even
though CC0 does not require it.

This is an independent project and is not affiliated with or endorsed by
Lichess.
