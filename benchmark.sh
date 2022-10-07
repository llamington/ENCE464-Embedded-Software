#!/usr/bin/sh

hyperfine --export-json benchmark.json \
--parameter-scan num_threads 1 7 \
'OMP_NUM_THREADS={num_threads} ./poisson -n 201 -i 300'

python benchmark_plot.py benchmark.json -o benchmark.png