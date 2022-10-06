#include "poisson_solver.hpp"
#include "util.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <omp.h>

PoissonSolver::PoissonSolver(int n,
                             const std::vector<double> &source,
                             int iterations,
                             int threads,
                             float delta,
                             bool debug)
    : n(n),
      source(source),
      iterations(iterations),
      delta(delta),
      threads(threads),
      curr(new std::vector<double>(n * n * n, 0)),
      next(new std::vector<double>(n * n * n)) {}

std::vector<double> *PoissonSolver::solve(void)
{
  auto time_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel shared(curr, next, source, delta, n, iterations)
  for (int iter = 0; iter < iterations; iter++)
  {
#pragma omp for schedule(static)
    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < n; j++)
      {
        for (int k = 0; k < n; k++)
        {
          double v = 0;

          if (i == 0)
            v += 2 * (*curr)[TENSOR_IDX(1, j, k, n)];
          else if (i == n - 1)
            v += 2 * (*curr)[TENSOR_IDX(n - 2, j, k, n)];
          else
          {
            v += (*curr)[TENSOR_IDX(i - 1, j, k, n)];
            v += (*curr)[TENSOR_IDX(i + 1, j, k, n)];
          }

          if (j == 0)
            v += 2 * (*curr)[TENSOR_IDX(i, 1, k, n)];
          else if (j == n - 1)
            v += 2 * (*curr)[TENSOR_IDX(i, n - 2, k, n)];
          else
          {
            v += (*curr)[TENSOR_IDX(i, j - 1, k, n)];
            v += (*curr)[TENSOR_IDX(i, j + 1, k, n)];
          }

          if (k == 0)
            v += 2 * (*curr)[TENSOR_IDX(i, j, 1, n)];
          else if (k == n - 1)
            v += 2 * (*curr)[TENSOR_IDX(i, j, n - 2, n)];
          else
          {
            v += (*curr)[TENSOR_IDX(i, j, k - 1, n)];
            v += (*curr)[TENSOR_IDX(i, j, k + 1, n)];
          }

          v -= delta * delta * source[TENSOR_IDX(i, j, k, n)];
          v /= 6;
          (*next)[TENSOR_IDX(i, j, k, n)] = v;
        }
      }
    }

#pragma omp single
    {
      std::vector<double> *temp = curr;
      curr = next;
      next = temp;
    }
  }

  auto time_stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time_stop - time_start);
  std::cout << "Duration: " << duration.count() << std::endl;

  return curr;
}