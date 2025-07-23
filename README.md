# Lichess Puzzle Presolver

This command-line tool presolves Lichess puzzles by running Stockfish analysis on each puzzle position. It reads a CSV of puzzles, evaluates each position and the position after the first move, and outputs a new CSV with Stockfish evaluations and best moves.

## Features
- Reads puzzles from a CSV file (must include `FEN` and `Moves` columns).
- Runs Stockfish to:
  - Evaluate the current position.
  - Find the best move.
  - Evaluate the position after the first move in the solution.
- Outputs a CSV with the original columns plus:
  - `stockfish_eval` (evaluation of the original position)
  - `stockfish_bestmove` (Stockfish's best move)
  - `stockfish_eval_after_first_move` (evaluation after the first move)
- All evaluations are **from White's perspective**.

## Requirements
- Python 3.7+
- [python-chess](https://pypi.org/project/python-chess/):
  ```sh
  pip install python-chess
  ```
- Stockfish engine installed and available in your WSL/Linux environment's PATH, or specify the path with `--stockfish-path`.

## Usage

```sh
python presolve_lichess_puzzles.py <input_csv> <output_csv> [--depth N] [--stockfish-path /path/to/stockfish]
```

- `<input_csv>`: Path to the input CSV file (must have `FEN` and `Moves` columns).
- `<output_csv>`: Path to the output CSV file.
- `--depth N`: (Optional) Stockfish search depth (default: 20).
- `--stockfish-path`: (Optional) Path to Stockfish binary (default: `stockfish`).

**Example:**
```sh
python presolve_lichess_puzzles.py puzzles/example.csv puzzles/example_out.csv --depth 20
```

## Stockfish Evaluation Handling
- **All evaluations are normalized to White's perspective:**
  - If the position is good for White, the evaluation is positive.
  - If the position is good for Black, the evaluation is negative.
  - This is true for both the original position and after the first move.
- The program automatically flips the sign of Stockfish's evaluation if the side to move in the FEN is Black.

## Mate Situations
- If Stockfish finds a forced mate, the evaluation is set to:
  - `+10000` if White is delivering mate (mate in N for White, or Black is being mated)
  - `-10000` if White is being mated (mate in N for Black, or White is being mated)
- This convention is used for both the original and after-move evaluations.
- The actual number of moves to mate is not preserved in the output; only the decisive result is indicated by ±10000.

## Notes
- The input CSV must have at least the columns `FEN` and `Moves` (case-sensitive).
- The script must be run in a Linux/WSL environment where Stockfish is available.

## License
MIT 