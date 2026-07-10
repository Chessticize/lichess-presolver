# Dataset provenance and quality

## Published snapshot

The retained corpus derives from a local copy of the cumulative Lichess puzzle
database downloaded on July 24, 2025. The upstream file did not encode a release
date, so the content hash—not a guessed day—is the durable snapshot identifier.

| Property | Value |
| --- | --- |
| Source | `https://database.lichess.org/lichess_db_puzzle.csv.zst` |
| Puzzle rows | 4,824,507 |
| Source CSV SHA-256 | `1b4e3b80d95b17798224c3edad7cfc0e9a599f71f1c3d2128b211dc812d5f2f6` |
| Split method | Eight-way round-robin |
| Retained analysis depth | 20 |
| Historical Stockfish build | Not recorded |

The exact engine build was not captured by the 2025 scripts. That is a real
reproducibility limitation; the cleaned presolver now emits a metadata sidecar
so future snapshots retain this information.

## Why depth 20 was retained

Two complete local outputs existed:

- `presolved-depth16`, written July 25, 2025; and
- `presolved`, written July 30, 2025.

Git commit `98dec03050ec03c512cb1e60a540cc4b231617b7` introduced the optimized
solver with a default depth of 16, despite help text that incorrectly said 20.
The later working-tree change corrected the default to 20 before `presolved`
was generated. File timestamps, equal input rows, and the code history therefore
identify the two corpora as depth 16 and depth 20 respectively.

The complete streaming comparison found:

| Metric | Result |
| --- | ---: |
| Paired rows | 4,824,507 |
| Missing, invalid, or source-mismatched rows | 0 |
| Identical best moves | 4,025,422 (83.4370%) |
| Different best moves | 799,085 |
| Median absolute delta, source-position evaluation | 13 cp |
| 95th percentile absolute delta, source-position evaluation | 111 cp |
| Median absolute delta, after-first-move evaluation | 18 cp |
| 95th percentile absolute delta, after-first-move evaluation | 176 cp |

To distinguish the corpora rather than merely measure disagreement, 128
best-move disagreements were selected with a deterministic reservoir sample
(`seed=20250725`) and re-evaluated with locally installed Stockfish 18 at depth
22:

| Reference comparison | Depth 16 | Later depth 20 |
| --- | ---: | ---: |
| Best move matched reference | 37 / 128 | **69 / 128** |
| Mean absolute error, source position | 39.709 cp | **22.142 cp** |
| Mean absolute error, after first move | 215.568 cp | **172.481 cp** |

The remaining 22 sampled best moves matched neither historical output. Because
the reference used a newer engine and is still depth-limited, this is a
comparative check rather than ground truth. Nevertheless, the generation
history, greater nominal depth, and all three reference metrics agree: the
depth-16 corpus is not better. Only the later depth-20 corpus is prepared as a
downloadable release.

The tracked manifests contain SHA-256 checksums for both the
[compressed release assets](../checksums/dataset-2025-07-depth20.sha256) and the
[decompressed CSV shards](../checksums/dataset-2025-07-depth20-raw.sha256).

Reproduce the audit with:

```sh
python3 tools/compare_outputs.py presolved-depth16 presolved \
  --sample-size 128 \
  --seed 20250725 \
  --reference-depth 22 \
  --stockfish /path/to/stockfish
```

## Data semantics and limitations

- All original Lichess puzzle columns are preserved without modification.
- Evaluations are from White's point of view.
- Mate scores are collapsed to `+10000` or `-10000`; mate distance is lost.
- `stockfish_bestmove` describes the source FEN, before the first listed move.
- The data is an historical snapshot and should not be confused with Lichess's
  continuously updated current export.
- A fixed depth is not a fixed node count or wall-clock effort across positions.
- The historical run did not save engine identity, NNUE network, hardware, or
  per-position node counts.

## Licensing and attribution

Lichess states that its database exports are released under CC0 and may be
downloaded, modified, and redistributed. The source and derived release assets
in this repository are provided under [CC0 1.0](../DATA_LICENSE.md).

Source: [Lichess open database](https://database.lichess.org/#puzzles).
