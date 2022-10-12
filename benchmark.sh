#!/usr/bin/sh

hyperfine --export-json benchmark.json \
--parameter-scan num_threads 1 8 \
'./poisson -n 201 -i 300 -t {num_threads}'

python benchmark_plot.py benchmark.json -o benchmark.png