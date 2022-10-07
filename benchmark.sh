#!/usr/bin/sh

hyperfine --export-json benchmark.json \
--parameter-scan num_threads 2 7 \
'./poisson -n 51 -i 300 -t {num_threads}'

python benchmark_plot.py benchmark.json -o benchmark.png