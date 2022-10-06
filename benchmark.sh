#!/usr/bin/sh

hyperfine --export-json benchmark.json \
--parameter-scan num_threads 1 7 \
'OMP_NUM_THREADS={num_threads} ./poisson -n 301 -i 30'

source venv/bin/activate

python benchmark_plot.py benchmark.json -o benchmark.png