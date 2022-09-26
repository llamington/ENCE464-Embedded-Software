import matplotlib.pyplot as plt
import numpy as np
import sys


prog_out = sys.stdin.read()
prog_out = prog_out.split("Result:\n")[1]
lines = prog_out.splitlines()

n = len(lines)
mat = np.empty((n, n))

for i, line in enumerate(lines):
    mat[i] = [float(val) for val in line.split()]

fig, ax = plt.subplots()
img = ax.matshow(mat)
fig.colorbar(img)
fig.savefig("poisson.png")

