import csv
import math
from typing import Optional

def split_csv(input_csv_path: str, num_splits: int, output_prefix: str = "split_") -> None:
    """
    Splits a CSV file into `num_splits` smaller files, preserving the header.
    Distributes rows in a round-robin fashion: row 1 to file 1, row 2 to file 2, ..., row N to file N, row N+1 to file 1, etc.
    Output files will be named as {output_prefix}1.csv, {output_prefix}2.csv, ...
    """
    if num_splits < 1:
        raise ValueError("num_splits must be at least 1")

    # Open all output files and writers
    out_files = []
    writers = []
    try:
        # Read header
        with open(input_csv_path, newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader)

            # Prepare output files and write headers
            for i in range(num_splits):
                out_path = f"{output_prefix}{i+1}.csv"
                f = open(out_path, 'w', newline='', encoding='utf-8')
                writer = csv.writer(f)
                writer.writerow(header)
                out_files.append(f)
                writers.append(writer)

            # Distribute rows round-robin
            for idx, row in enumerate(reader):
                file_idx = idx % num_splits
                writers[file_idx].writerow(row)
    finally:
        for f in out_files:
            f.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Split a CSV file into multiple parts.")
    parser.add_argument("input_csv_path", help="Path to the input CSV file.")
    parser.add_argument("num_splits", type=int, help="Number of output files to create.")
    parser.add_argument("--output_prefix", default="split_", help="Prefix for output files (default: split_)")
    args = parser.parse_args()
    split_csv(args.input_csv_path, args.num_splits, args.output_prefix)

if __name__ == "__main__":
    main() 