# Arrow Duel route metrics

## Metric contract

The root position is excluded from the difficulty count because Arrow Duel
already presents exactly two candidates. The analyzer constrains Stockfish to
the stored top move, follows the opponent's best reply, and then measures how
many later puzzle-side turns have exactly one good continuation.

`top_move_unique_plies` is the compact raw value. It includes the displayed
top move and intervening opponent replies because that representation preserves
terminal timing in one four-bit value. The report derives
`unique_followup_moves` as:

```text
floor((top_move_unique_plies - 1) / 2)
```

The displayed top move is therefore excluded from `unique_followup_moves`.
With the default ten-ply window, values nine and ten establish exactly four
forced follow-ups. Code eleven establishes at least four and is reported in the
same censored `4+` bucket.

A continuation is good when it is no more than 100 centipawns below the best
MultiPV line from the original puzzle side's perspective. When the best line is
a forced mate, only alternatives with the same mate distance qualify. The
current contract uses MultiPV 2 and starts a fresh UCI game for every puzzle.
The second-best line is sufficient: if it qualifies, the route is not unique;
if it does not qualify, no lower-ranked line can qualify.

These values are qualified by Stockfish version, search depth, MultiPV count,
evaluation tolerance, and route window. They are not engine-independent puzzle
facts. Every output records that analysis identity and the exact input SHA-256.

## MultiPV 5 search-depth baseline

The benchmark was run on August 3, 2026 with Stockfish 18. The versioned
binary-v1 sample has SHA-256
`0676803ffb4e67fe2f47c60eacc78f9321896afacbaa3d993bbe69485f5a1b10`,
2,001 Core Pack rows, and 1,990 Arrow Duel-eligible rows. Ratings range from
600 to 2,200 with a mean of 1,434.4. Each run used ten independent
single-threaded workers, 4 MB hash per worker, the former MultiPV 5 contract,
a 100-centipawn tolerance, a ten-ply route window, and the same machine.

| Depth | Wall time | Milliseconds per input | Relative to depth 10 | Linear 1,392,969-input estimate |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 16.00 s | 8.04 ms | 1.0x | 3.11 h |
| 16 | 367.62 s | 184.73 ms | 23.0x | 71.5 h |
| 20 | 2,508.41 s | 1,260.51 ms | 156.8x | 487.8 h (20.3 d) |

The full-pack estimates are simple linear projections. Depth-20 positions had
a pronounced long tail, so the estimate is planning guidance rather than a
runtime guarantee.

### Follow-up distributions

| Depth | 0 | 1 | 2 | 3 | 4+ | At least 1 | At least 2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 1,439 | 338 | 119 | 54 | 40 | 551 (27.69%) | 213 (10.70%) |
| 16 | 1,432 | 324 | 133 | 57 | 44 | 558 (28.04%) | 234 (11.76%) |
| 20 | 1,406 | 327 | 134 | 62 | 61 | 584 (29.35%) | 257 (12.91%) |

Aggregate counts hide substantial row-level churn because increases and
decreases partly cancel:

| Comparison | Raw ply code changed | Follow-up bucket changed | Crossed at-least-1 boundary | Crossed at-least-2 boundary |
| --- | ---: | ---: | ---: | ---: |
| Depth 10 to 16 | 376 (18.89%) | 374 (18.79%) | 283 (14.22%) | 173 (8.69%) |
| Depth 16 to 20 | 273 (13.72%) | 265 (13.32%) | 188 (9.45%) | 123 (6.18%) |
| Depth 10 to 20 | 440 (22.11%) | 434 (21.81%) | 315 (15.83%) | 216 (10.85%) |

For the at-least-one boundary, depth 10 to 16 moved 145 rows into the forced
set and 138 out. Depth 16 to 20 moved 107 in and 81 out. Only 1,475 rows
(74.12%) kept the same derived follow-up bucket at all three depths.

### Interpretation

Depth 10 is inexpensive enough for a full-pack exploratory pass but is not
stable enough to treat individual row labels as durable facts. Depth 16 is
roughly 23 times slower and still differs materially from depth 20. A production
classification must therefore keep the engine/depth identity and choose an
explicit accuracy policy before a multi-day full-pack run.

## MultiPV 2 production benchmark

The same sample, machine, Stockfish version, worker layout, route window, and
score threshold were rerun with MultiPV 2. This is the current production
contract because only the best and second-best moves are required to decide
uniqueness.

| Depth | Wall time | Milliseconds per input | Speedup over MultiPV 5 | Linear 1,392,969-input estimate |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 7.99 s | 4.02 ms | 2.00x | 1.55 h |
| 16 | 188.85 s | 94.90 ms | 1.95x | 36.7 h |
| 20 | 1,389.97 s | 698.48 ms | 1.80x | 270.3 h (11.3 d) |

### MultiPV 2 follow-up distributions

| Depth | 0 | 1 | 2 | 3 | 4+ | At least 1 | At least 2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 1,422 | 367 | 106 | 54 | 41 | 568 (28.54%) | 201 (10.10%) |
| 16 | 1,406 | 338 | 142 | 57 | 47 | 584 (29.35%) | 246 (12.36%) |
| 20 | 1,399 | 319 | 136 | 68 | 68 | 591 (29.70%) | 272 (13.67%) |

Changing MultiPV changes Stockfish's fixed-depth search allocation, so the
MultiPV 2 result is not guaranteed to reproduce the former MultiPV 5 label at
the same depth. The logical uniqueness rule is unchanged, but the
engine-qualified analysis identity is different:

| Depth | Raw ply code changed | Follow-up bucket changed | Crossed at-least-1 boundary |
| ---: | ---: | ---: | ---: |
| 10 | 193 (9.70%) | 192 (9.65%) | 111 (5.58%) |
| 16 | 147 (7.39%) | 145 (7.29%) | 90 (4.52%) |
| 20 | 144 (7.24%) | 138 (6.93%) | 59 (2.96%) |

Depth sensitivity remains material under MultiPV 2:

| Comparison | Raw ply code changed | Follow-up bucket changed | Crossed at-least-1 boundary |
| --- | ---: | ---: | ---: |
| Depth 10 to 16 | 390 (19.60%) | 385 (19.35%) | 284 (14.27%) |
| Depth 16 to 20 | 291 (14.62%) | 284 (14.27%) | 203 (10.20%) |
| Depth 10 to 20 | 446 (22.41%) | 442 (22.21%) | 321 (16.13%) |

Only 1,461 rows (73.42%) kept the same follow-up bucket at all three MultiPV 2
depths. MultiPV 2 approximately halves runtime without making shallow fixed
depth labels converge. Production outputs must continue to record both depth
and MultiPV in their identity.
