import matplotlib.pyplot as plt
import numpy as np
import sys


lines = sys.stdin.readlines()
n = len(lines)
mat = np.empty((n, n))

for i, line in enumerate(lines):
    mat[i] = [float(val) for val in line.split()]

fig, ax = plt.subplots()
img = ax.matshow(mat)
fig.colorbar(img)
fig.savefig("poisson.png")

