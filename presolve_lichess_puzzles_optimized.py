#!/usr/bin/env python3
import argparse
import csv
import subprocess
import sys
import os
from typing import Tuple, List
import threading
import queue


class StockfishEngine:
    """A persistent Stockfish engine that can be reused for multiple evaluations."""

    def __init__(self, stockfish_path: str = "stockfish"):
        self.stockfish_path = stockfish_path
        self.proc = None
        self.lock = threading.Lock()
        # Reusable chess board to avoid repeated object creation
        try:
            import chess
            self.board = chess.Board()
        except ImportError:
            print("python-chess is required. Install with 'pip install python-chess'", file=sys.stderr)
            sys.exit(1)
        self._start_engine()
    
    def _start_engine(self):
        """Start the Stockfish process and initialize UCI."""
        cmd = [self.stockfish_path]
        self.proc = subprocess.Popen(
            cmd, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            bufsize=0
        )
        
        # Initialize and optimize
        self._send("setoption name Threads value 1")
        self._send("setoption name Hash value 128")
        self._send("setoption name UCI_Variant value chess")
        self._send("setoption name UCI_Chess960 value false")

        # UCI initialization
        self._send("uci")
        self._wait_for("uciok")
        self._send("isready")
        self._wait_for("readyok")
    
    def _send(self, cmd: str):
        """Send a command to Stockfish."""
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()
    
    def _wait_for(self, expected: str):
        """Wait for a specific response from Stockfish."""
        while True:
            line = self.proc.stdout.readline().strip()
            if line == expected:
                break
    
    def evaluate(self, fen: str, depth: int) -> Tuple[float, str]:
        """
        Evaluate a position and return (evaluation, bestmove).
        Thread-safe.
        """
        with self.lock:
            return self._evaluate_unsafe(fen, depth)

    def get_side_to_move(self, fen: str) -> bool:
        """
        Get side to move from FEN using reusable board.
        Returns True if white to move, False if black to move.
        Thread-safe.
        """
        with self.lock:
            self.board.set_fen(fen)
            return self.board.turn  # True for white, False for black

    def apply_move_to_fen(self, fen: str, move: str) -> str:
        """
        Apply a UCI move to a FEN string using reusable board.
        Thread-safe.
        """
        with self.lock:
            try:
                self.board.set_fen(fen)
                self.board.push_uci(move)
                result_fen = self.board.fen()
                return result_fen
            except Exception as e:
                print(f"Error applying move {move} to FEN {fen}: {e}", file=sys.stderr)
                sys.exit(1)
    
    def _evaluate_unsafe(self, fen: str, depth: int) -> Tuple[float, str]:
        """Internal evaluation method (not thread-safe)."""
        # Set position
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        
        eval_cp = None
        eval_mate = None
        bestmove = None
        
        while True:
            line = self.proc.stdout.readline()
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
        
        # Convert evaluation
        if eval_mate is not None:
            eval_score = 10000 if eval_mate > 0 else -10000
        elif eval_cp is not None:
            eval_score = float(eval_cp)
        else:
            eval_score = None
            
        return eval_score, bestmove
    
    def close(self):
        """Close the Stockfish engine."""
        if self.proc:
            try:
                self._send("quit")
                self.proc.stdin.close()
                self.proc.stdout.close()
                self.proc.stderr.close()
                self.proc.wait(timeout=5)
            except:
                self.proc.kill()
            self.proc = None





def process_puzzles(input_csv: str, output_csv: str, depth: int, stockfish_path: str):
    """Process puzzles using a persistent Stockfish engine."""
    engine = StockfishEngine(stockfish_path)
    
    try:
        with open(input_csv, newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames + [
                'stockfish_eval', 'stockfish_bestmove', 'stockfish_eval_after_first_move'
            ]
            
            with open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for i, row in enumerate(reader, 1):
                    if i % 10 == 0:
                        import datetime
                        now = datetime.datetime.now()
                        print(f"{now.strftime('%H:%M:%S')} Processing puzzle {i}", file=sys.stderr)
                    
                    fen = row['FEN']
                    moves = row['Moves']
                    
                    # Evaluate initial position
                    eval1, bestmove = engine.evaluate(fen, depth)

                    # Flip sign if side to move is black
                    if not engine.get_side_to_move(fen) and eval1 is not None:  # False = black to move
                        eval1 = -eval1

                    # Get first move and evaluate position after it
                    first_move = moves.split()[0].split(',')[0]
                    fen_after = engine.apply_move_to_fen(fen, first_move)
                    eval2, _ = engine.evaluate(fen_after, depth)

                    # Flip sign if side to move is black after the move
                    if not engine.get_side_to_move(fen_after) and eval2 is not None:  # False = black to move
                        eval2 = -eval2
                    
                    row['stockfish_eval'] = eval1
                    row['stockfish_bestmove'] = bestmove
                    row['stockfish_eval_after_first_move'] = eval2
                    writer.writerow(row)
                    
    finally:
        engine.close()


def main():
    parser = argparse.ArgumentParser(description="Presolve Lichess puzzles using Stockfish (optimized).")
    parser.add_argument('input_csv', help="Input CSV file with puzzles (must have FEN and Moves columns)")
    parser.add_argument('output_csv', help="Output CSV file with results")
    parser.add_argument('--depth', type=int, default=16, help="Stockfish search depth (default: 20)")
    parser.add_argument('--stockfish-path', default='stockfish', help="Path to Stockfish binary (default: stockfish)")
    args = parser.parse_args()
    process_puzzles(args.input_csv, args.output_csv, args.depth, args.stockfish_path)

if __name__ == '__main__':
    main()
