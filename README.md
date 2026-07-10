# Lichess Puzzle Presolver

Lichess Puzzle Presolver adds reproducible Stockfish evaluations to a snapshot
of the [Lichess puzzle database](https://database.lichess.org/#puzzles). It can
process one CSV, split a large snapshot, or run several split files in parallel.

The repository also distributes a presolved July 2025 snapshot as compressed
[GitHub Release assets](https://github.com/Chessticize/lichess-presolve/releases).
Only the later **depth-20** corpus is distributed. A complete comparison found
that the older `presolved-depth16` corpus was not the better-quality set; see
[Dataset provenance and quality](docs/DATASET.md).

## Download the presolved data

Download all eight independently decompressible assets from the latest dataset
release:

```sh
gh release download dataset-2025-07-depth20 \
  --repo Chessticize/lichess-presolve
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

See [Usage guide](docs/USAGE.md) for installation, resource sizing, CSV
semantics, verification, and troubleshooting.

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
