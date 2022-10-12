#!/usr/bin/env python
"""Script for generating poisson solver diagrams"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import re


prog_out = sys.stdin.read()
sections = prog_out.split("Result:\n")
thread_prof_str = sections[0]
matrix_str = sections[1]
matrix_lines = matrix_str.splitlines()

thread_execution = {}

pattern = re.compile("Thread ([0-9]+): ([0-9]+)")
for line in thread_prof_str.splitlines():
    match = pattern.match(line)
    if match:
        thread_num, execution = match.group(1, 2)
        thread_execution[int(thread_num)] = thread_execution.get(thread_num, []) + [int(execution)]

for k, v in thread_execution.items():
    thread_execution[k] = sum(v) / len(v)


fig, ax = plt.subplots()
ax.bar(thread_execution.keys(), thread_execution.values())
fig.savefig("thread_timings.png")

n = len(matrix_lines)
mat = np.empty((n, n))

for i, line in enumerate(matrix_lines):
    mat[i] = np.array([float(val) for val in line.split()])

fig, ax = plt.subplots()
img = ax.matshow(mat)
fig.colorbar(img)
fig.savefig("poisson.png")

