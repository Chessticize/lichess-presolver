#!/bin/bash

mkdir -p presolved

find puzzles -name 'split_*.csv' | xargs -I{} -P 10 bash -c '
    infile="{}"
    fname=$(basename "$infile")
    outfile="presolved/$fname"
    echo "Processing $infile -> $outfile"
    python3 presolve_lichess_puzzles.py "$infile" "$outfile"
'
