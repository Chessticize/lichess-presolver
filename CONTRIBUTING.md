# Contributing

Contributions are welcome. Keep changes focused and include tests for changed
public behavior.

Before opening a pull request:

```sh
python3 -m compileall -q presolve_lichess_puzzles.py \
  classify_arrow_duel_routes.py solve_parallel.py split_csv_util.py tools tests
python3 -m unittest discover -s tests -v
git diff --check
```

The integration test intentionally starts a real Stockfish 18 process. Ensure
the `stockfish` executable on `PATH` reports Stockfish 18 before running the
suite, because the route regression values are engine-qualified. Do not replace
it with a mock of internal behavior. Tests for failure modes may use a small
maintained fake UCI executable when a real engine cannot produce the condition
deterministically.

Do not commit Lichess snapshots, generated CSVs, compressed release assets, or
partial outputs. Publish large datasets through GitHub Releases and include a
SHA-256 manifest plus provenance notes.
