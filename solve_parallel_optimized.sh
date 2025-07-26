#!/bin/bash

mkdir -p presolved

echo "Starting optimized parallel processing with 10 workers..."
echo "Processing $(find puzzles -name 'split_*.csv' | wc -l) files"

find puzzles -name 'split_*.csv' | xargs -I{} -P 8 bash -c '
    infile="{}"
    fname=$(basename "$infile")
    outfile="presolved/$fname"
    echo "Processing $infile -> $outfile"
    python3 presolve_lichess_puzzles_optimized.py "$infile" "$outfile"
'

echo "All files processed!"
