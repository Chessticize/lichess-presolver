!/bin/bash

mkdir -p presolved

for infile in puzzles/split_*.csv; do
    fname=$(basename "$infile")
    outfile="presolved/$fname"
    echo "Processing $infile -> $outfile"
    python3 presolve_lichess_puzzles.py "$infile" "$outfile"
done

