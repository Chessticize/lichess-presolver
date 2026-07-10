# Usage guide

## Install Stockfish

Install Python 3.10 or newer and a Stockfish executable. Examples:

```sh
# macOS
brew install stockfish

# Debian or Ubuntu
sudo apt-get install stockfish
```

If the executable is not on `PATH`, pass its full location with
`--stockfish-path`.

## Download a Lichess puzzle snapshot

Lichess publishes one cumulative puzzle CSV compressed with Zstandard:

```sh
curl -LO https://database.lichess.org/lichess_db_puzzle.csv.zst
unzstd lichess_db_puzzle.csv.zst
```

The upstream schema and puzzle semantics are documented on the
[Lichess open database page](https://database.lichess.org/#puzzles).

## Process one CSV

```sh
python3 presolve_lichess_puzzles.py INPUT.csv OUTPUT.csv \
  --depth 20 \
  --stockfish-path stockfish \
  --threads 1 \
  --hash-mb 128
```

The input must contain the ten standard Lichess puzzle columns. The command
uses one persistent Stockfish process for the entire file and writes through a
`.partial` file so an interrupted run does not replace a completed output.

By default it also writes `OUTPUT.csv.metadata.json`. The sidecar records the
engine name, analysis settings, row count, timestamps, and input/output hashes.
Use `--no-metadata` only when the extra provenance file is undesirable.

## Understand the positions

Lichess defines `FEN` as the position *before* the opponent plays the first move
listed in `Moves`. The player-facing puzzle begins after that first move, and
the second move starts the solution.

This repository therefore stores:

- `stockfish_eval`: the source position before the opponent's recorded move;
- `stockfish_bestmove`: Stockfish's preferred move instead of that recorded
  opponent move; and
- `stockfish_eval_after_first_move`: the player-facing puzzle position.

Scores are integer centipawns from White's point of view. Positive is favorable
for White and negative is favorable for Black. Mate scores collapse to
`+10000` or `-10000`.

## Split and process in parallel

Split a large CSV round-robin so every output has nearly the same row count:

```sh
python3 split_csv_util.py lichess_db_puzzle.csv 8 \
  --output-prefix puzzles/split_
```

Run one Stockfish process per split:

```sh
python3 solve_parallel.py \
  --input-dir puzzles \
  --output-dir presolved-depth20 \
  --pattern 'split_*.csv' \
  --depth 20 \
  --jobs 8 \
  --threads-per-job 1 \
  --hash-mb 128
```

Approximate engine memory is `jobs × hash-mb`, plus Stockfish and Python
overhead. Start with one thread per job; increasing both jobs and threads can
oversubscribe the machine and make the run slower.

Existing outputs are not replaced unless `--overwrite` is supplied. If one
worker fails, the command reports every failed input and returns a non-zero exit
status.

## Compare two generated corpora

The repository includes the streaming audit used for the published dataset:

```sh
python3 tools/compare_outputs.py LEFT_DIR RIGHT_DIR
```

To check a deterministic sample of best-move disagreements against a deeper
local Stockfish run:

```sh
python3 tools/compare_outputs.py LEFT_DIR RIGHT_DIR \
  --sample-size 128 \
  --reference-depth 22 \
  --stockfish /path/to/stockfish
```

The deeper comparison is useful evidence, not an absolute ground truth: engine
versions can disagree, and the historical snapshot did not record its exact
Stockfish build.

## Download and verify release assets

Each release CSV split is compressed separately, so consumers can download only
the shards they need:

```sh
gh release download dataset-2025-07-depth20 \
  --repo Chessticize/lichess-presolve
shasum -a 256 -c SHA256SUMS
unzstd lichess-puzzles-presolved-depth20-2025-07-split-1.csv.zst
```

Use `zstdcat FILE.csv.zst` to stream a shard without first writing the
uncompressed CSV.

## Run the tests

```sh
python3 -m unittest discover -s tests -v
```

The CLI integration test starts the real local Stockfish process and exercises
the public CSV interface. It is skipped only when Stockfish is unavailable.
