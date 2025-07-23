#!/usr/bin/env python3
import argparse
import csv
import subprocess
import sys
import os
from typing import Tuple, List


def run_stockfish(fen: str, depth: int, stockfish_path: str = "stockfish") -> Tuple[float, str]:
    """
    Run Stockfish on the given FEN at the given depth.
    Returns (evaluation, bestmove), evaluation is from white's POV in centipawns (float, mate as 10000 or -10000).
    """
    # Prepare the Stockfish command
    cmd = [stockfish_path]
    # Start Stockfish process
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    def send(cmd):
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()
    
    # UCI init"
    send("uci")
    # Wait for 'uciok'
    while True:
        line = proc.stdout.readline()
        if line.strip() == "uciok":
            break
    send("isready")
    while True:
        line = proc.stdout.readline()
        if line.strip() == "readyok":
            break
    # Set position
    send(f"position fen {fen}")
    send(f"go depth {depth}")
    eval_cp = None
    eval_mate = None
    bestmove = None
    while True:
        line = proc.stdout.readline()
        if line.startswith("info"):
            if "score cp" in line:
                try:
                    eval_cp = int(line.split("score cp ")[1].split()[0])
                except Exception:
                    pass
            elif "score mate" in line:
                try:
                    eval_mate = int(line.split("score mate ")[1].split()[0])
                except Exception:
                    pass
        elif line.startswith("bestmove"):
            bestmove = line.split()[1]
            break
    # Clean up
    send("quit")
    proc.stdin.close()
    proc.stdout.close()
    proc.stderr.close()
    proc.wait()
    # Prefer mate score if present
    if eval_mate is not None:
        eval_score = 10000 if eval_mate > 0 else -10000
    elif eval_cp is not None:
        eval_score = float(eval_cp)
    else:
        eval_score = None
    return eval_score, bestmove


def apply_move_to_fen(fen: str, move: str) -> str:
    """
    Apply a UCI move to a FEN string. Uses python-chess if available, else fails.
    """
    try:
        import chess
        board = chess.Board(fen)
        board.push_uci(move)
        return board.fen()
    except ImportError:
        print("python-chess is required for move application. Install with 'pip install python-chess'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error applying move {move} to FEN {fen}: {e}", file=sys.stderr)
        sys.exit(1)


def process_puzzles(input_csv: str, output_csv: str, depth: int, stockfish_path: str):
    with open(input_csv, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + [
            'stockfish_eval', 'stockfish_bestmove', 'stockfish_eval_after_first_move'
        ]
        with open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for i, row in enumerate(reader, 1):
                fen = row['FEN']
                moves = row['Moves']
                print(f"Processing puzzle {i}", file=sys.stderr)
                eval1, bestmove = run_stockfish(fen, depth, stockfish_path)
                # Flip sign if side to move is black
                try:
                    import chess
                    board = chess.Board(fen)
                    if board.turn == chess.BLACK and eval1 is not None:
                        eval1 = -eval1
                except ImportError:
                    print("python-chess is required for move application. Install with 'pip install python-chess'", file=sys.stderr)
                    sys.exit(1)
                # Get first move from moves (space or comma separated)
                first_move = moves.split()[0].split(',')[0]
                fen_after = apply_move_to_fen(fen, first_move)
                eval2, _ = run_stockfish(fen_after, depth, stockfish_path)
                # Flip sign if side to move is black after the move
                board_after = chess.Board(fen_after)
                if board_after.turn == chess.BLACK and eval2 is not None:
                    eval2 = -eval2
                row['stockfish_eval'] = eval1
                row['stockfish_bestmove'] = bestmove
                row['stockfish_eval_after_first_move'] = eval2
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Presolve Lichess puzzles using Stockfish.")
    parser.add_argument('input_csv', help="Input CSV file with puzzles (must have FEN and Moves columns)")
    parser.add_argument('output_csv', help="Output CSV file with results")
    parser.add_argument('--depth', type=int, default=20, help="Stockfish search depth (default: 20)")
    parser.add_argument('--stockfish-path', default='stockfish', help="Path to Stockfish binary (default: stockfish)")
    args = parser.parse_args()
    process_puzzles(args.input_csv, args.output_csv, args.depth, args.stockfish_path)

if __name__ == '__main__':
    main() 