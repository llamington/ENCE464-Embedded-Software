#include "poisson_solver.hpp"
#include "util.hpp"
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <omp.h>

PoissonSolver::PoissonSolver(int n,
                             const std::vector<float> &source,
                             int iterations,
                             int threads,
                             float delta,
                             bool debug)
    : n(n),
      source(source),
      iterations(iterations),
      threads(threads),
      delta(delta),
      curr(new std::vector<float>(n * n * n, 0)),
      next(new std::vector<float>(n * n * n)) {}

std::vector<float> *PoissonSolver::solve(void)
{
  auto time_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel shared(curr, next, source, delta, n, iterations)
  for (uint16_t iter = 0; iter < iterations; iter++)
  {
    float v = 0;
    int i, j, k;
#pragma omp sections nowait
    {
      // Outer Faces
#pragma omp section
      {
        for (i = 1; i < n - 1; i++)
        {
          for (j = 1; j < n - 1; j++)
          {
            // Zeroth face
            v = 0;
            v += (*curr)[TENSOR_IDX(i - 1, j, 0, n)];
            v += (*curr)[TENSOR_IDX(i, j - 1, 0, n)];
            v += 2 * (*curr)[TENSOR_IDX(i, j, 1, n)];
            v += (*curr)[TENSOR_IDX(i, j + 1, 0, n)];
            v += (*curr)[TENSOR_IDX(i + 1, j, 0, n)];
            v -= delta * delta * source[TENSOR_IDX(i, j, 0, n)];
            v /= 6;
            (*next)[TENSOR_IDX(i, j, 0, n)] = v;

            // n-1th face
            v = 0;
            v += (*curr)[TENSOR_IDX(i - 1, j, n - 1, n)];
            v += (*curr)[TENSOR_IDX(i, j - 1, n - 1, n)];
            v += 2 * (*curr)[TENSOR_IDX(i, j, n - 2, n)];
            v += (*curr)[TENSOR_IDX(i, j + 1, n - 1, n)];
            v += (*curr)[TENSOR_IDX(i + 1, j, n - 1, n)];
            v -= delta * delta * source[TENSOR_IDX(i, j, n - 1, n)];
            v /= 6;
            (*next)[TENSOR_IDX(i, j, n - 1, n)] = v;
          }
        }
      }
#pragma omp section
      {
        for (i = 1; i < n - 1; i++)
        {
          for (k = 1; k < n - 1; k++)
          {
            // Zeroth face
            v = 0;
            v += (*curr)[TENSOR_IDX(i - 1, 0, k, n)];
            v += 2 * (*curr)[TENSOR_IDX(i, 1, k, n)];
            v += (*curr)[TENSOR_IDX(i, 0, k - 1, n)];
            v += (*curr)[TENSOR_IDX(i, 0, k + 1, n)];
            v += (*curr)[TENSOR_IDX(i + 1, 0, k, n)];
            v -= delta * delta * source[TENSOR_IDX(i, 0, k, n)];
            v /= 6;
            (*next)[TENSOR_IDX(i, 0, k, n)] = v;

            // n-1th face
            v = 0;
            v += (*curr)[TENSOR_IDX(i - 1, n - 1, k, n)];
            v += 2 * (*curr)[TENSOR_IDX(i, n - 2, k, n)];
            v += (*curr)[TENSOR_IDX(i, n - 1, k - 1, n)];
            v += (*curr)[TENSOR_IDX(i, n - 1, k + 1, n)];
            v += (*curr)[TENSOR_IDX(i + 1, n - 1, k, n)];
            v -= delta * delta * source[TENSOR_IDX(i, n - 1, k, n)];
            v /= 6;
            (*next)[TENSOR_IDX(i, n - 1, k, n)] = v;
          }
        }
      }
#pragma omp section
      {
        for (j = 1; j < n - 1; j++)
        {
          for (k = 1; k < n - 1; k++)
          {
            // Zeroth face
            v = 0;
            v += (*curr)[TENSOR_IDX(0, j - 1, k, n)];
            v += (*curr)[TENSOR_IDX(0, j, k - 1, n)];
            v += (*curr)[TENSOR_IDX(0, j, k + 1, n)];
            v += (*curr)[TENSOR_IDX(0, j + 1, k, n)];
            v += 2 * (*curr)[TENSOR_IDX(1, j, k, n)];
            v -= delta * delta * source[TENSOR_IDX(0, j, k, n)];
            v /= 6;
            (*next)[TENSOR_IDX(0, j, k, n)] = v;

            // n-1th face
            v = 0;
            v += 2 * (*curr)[TENSOR_IDX(n - 2, j, k, n)];
            v += (*curr)[TENSOR_IDX(n - 1, j - 1, k, n)];
            v += (*curr)[TENSOR_IDX(n - 1, j, k - 1, n)];
            v += (*curr)[TENSOR_IDX(n - 1, j, k + 1, n)];
            v += (*curr)[TENSOR_IDX(n - 1, j + 1, k, n)];
            v -= delta * delta * source[TENSOR_IDX(n - 1, j, k, n)];
            v /= 6;
            (*next)[TENSOR_IDX(n - 1, j, k, n)] = v;
          }
        }
      }
// Outer Edges
#pragma omp section
      {
        for (i = 1; i < n - 1; i++)
        {
          // (0,0)
          v = 0;
          v += (*curr)[TENSOR_IDX(i - 1, 0, 0, n)];
          v += 2 * (*curr)[TENSOR_IDX(i, 0, 1, n)];
          v += 2 * (*curr)[TENSOR_IDX(i, 1, 0, n)];
          v += (*curr)[TENSOR_IDX(i + 1, 0, 0, n)];
          v -= delta * delta * source[TENSOR_IDX(i, 0, 0, n)];
          v /= 6;
          (*next)[TENSOR_IDX(i, 0, 0, n)] = v;

          // (0, n-1)
          v = 0;
          v += (*curr)[TENSOR_IDX(i - 1, 0, n - 1, n)];
          v += 2 * (*curr)[TENSOR_IDX(i, 0, n - 2, n)];
          v += 2 * (*curr)[TENSOR_IDX(i, 1, n - 1, n)];
          v += (*curr)[TENSOR_IDX(i + 1, 0, n - 1, n)];
          v -= delta * delta * source[TENSOR_IDX(i, 0, n - 1, n)];
          v /= 6;
          (*next)[TENSOR_IDX(i, 0, n - 1, n)] = v;

          // (n-1, 0)
          v = 0;
          v += (*curr)[TENSOR_IDX(i - 1, n - 1, 0, n)];
          v += 2 * (*curr)[TENSOR_IDX(i, n - 2, 0, n)];
          v += 2 * (*curr)[TENSOR_IDX(i, n - 1, 1, n)];
          v += (*curr)[TENSOR_IDX(i + 1, n - 1, 0, n)];
          v -= delta * delta * source[TENSOR_IDX(i, n - 1, 0, n)];
          v /= 6;
          (*next)[TENSOR_IDX(i, n - 1, 0, n)] = v;

          // (n-1, n-1)
          v = 0;
          v += (*curr)[TENSOR_IDX(i - 1, n - 1, n - 1, n)];
          v += 2 * (*curr)[TENSOR_IDX(i, n - 2, n - 1, n)];
          v += 2 * (*curr)[TENSOR_IDX(i, n - 1, n - 2, n)];
          v += (*curr)[TENSOR_IDX(i + 1, n - 1, n - 1, n)];
          v -= delta * delta * source[TENSOR_IDX(i, n - 1, n - 1, n)];
          v /= 6;
          (*next)[TENSOR_IDX(i, n - 1, n - 1, n)] = v;
        }
      }
#pragma omp section
      {
        for (j = 1; j < n - 1; j++)
        {
          // (0,0)
          v = 0;
          v += (*curr)[TENSOR_IDX(0, j - 1, 0, n)];
          v += 2 * (*curr)[TENSOR_IDX(0, j, 1, n)];
          v += (*curr)[TENSOR_IDX(0, j + 1, 0, n)];
          v += 2 * (*curr)[TENSOR_IDX(1, j, 0, n)];
          v -= delta * delta * source[TENSOR_IDX(0, j, 0, n)];
          v /= 6;
          (*next)[TENSOR_IDX(0, j, 0, n)] = v;

          // (0, n-1)
          v = 0;
          v += (*curr)[TENSOR_IDX(0, j - 1, n - 1, n)];
          v += 2 * (*curr)[TENSOR_IDX(0, j, n - 2, n)];
          v += (*curr)[TENSOR_IDX(0, j + 1, n - 1, n)];
          v += 2 * (*curr)[TENSOR_IDX(1, j, n - 1, n)];
          v -= delta * delta * source[TENSOR_IDX(0, j, n - 1, n)];
          v /= 6;
          (*next)[TENSOR_IDX(0, j, n - 1, n)] = v;

          // (n-1, 0)
          v = 0;
          v += 2 * (*curr)[TENSOR_IDX(n - 2, j, 0, n)];
          v += (*curr)[TENSOR_IDX(n - 1, j - 1, 0, n)];
          v += 2 * (*curr)[TENSOR_IDX(n - 1, j, 1, n)];
          v += (*curr)[TENSOR_IDX(n - 1, j + 1, 0, n)];
          v -= delta * delta * source[TENSOR_IDX(n - 1, j, 0, n)];
          v /= 6;
          (*next)[TENSOR_IDX(n - 1, j, 0, n)] = v;

          // (n-1, n-1)
          v = 0;
          v += 2 * (*curr)[TENSOR_IDX(n - 2, j, n - 1, n)];
          v += (*curr)[TENSOR_IDX(n - 1, j - 1, n - 1, n)];
          v += 2 * (*curr)[TENSOR_IDX(n - 1, j, n - 2, n)];
          v += (*curr)[TENSOR_IDX(n - 1, j + 1, n - 1, n)];
          v -= delta * delta * source[TENSOR_IDX(n - 1, j, n - 1, n)];
          v /= 6;
          (*next)[TENSOR_IDX(n - 1, j, n - 1, n)] = v;
        }
      }
#pragma omp section
      {
        for (k = 1; k < n - 1; k++)
        {
          // (0,0)
          v = 0;
          v += (*curr)[TENSOR_IDX(0, 0, k - 1, n)];
          v += (*curr)[TENSOR_IDX(0, 0, k + 1, n)];
          v += 2 * (*curr)[TENSOR_IDX(0, 1, k, n)];
          v += 2 * (*curr)[TENSOR_IDX(1, 0, k, n)];
          v -= delta * delta * source[TENSOR_IDX(0, 0, k, n)];
          v /= 6;
          (*next)[TENSOR_IDX(0, 0, k, n)] = v;

          // (0, n-1)
          v = 0;
          v += 2 * (*curr)[TENSOR_IDX(0, n - 2, k, n)];
          v += (*curr)[TENSOR_IDX(0, n - 1, k - 1, n)];
          v += (*curr)[TENSOR_IDX(0, n - 1, k + 1, n)];
          v += 2 * (*curr)[TENSOR_IDX(1, n - 1, k, n)];
          v -= delta * delta * source[TENSOR_IDX(0, n - 1, k, n)];
          v /= 6;
          (*next)[TENSOR_IDX(0, n - 1, k, n)] = v;

          // (n-1, 0)
          v = 0;
          v += 2 * (*curr)[TENSOR_IDX(n - 2, 0, k, n)];
          v += (*curr)[TENSOR_IDX(n - 1, 0, k - 1, n)];
          v += (*curr)[TENSOR_IDX(n - 1, 0, k + 1, n)];
          v += 2 * (*curr)[TENSOR_IDX(n - 1, 1, k, n)];
          v -= delta * delta * source[TENSOR_IDX(n - 1, 0, k, n)];
          v /= 6;
          (*next)[TENSOR_IDX(n - 1, 0, k, n)] = v;

          // (n-1, n-1)
          v = 0;
          v += 2 * (*curr)[TENSOR_IDX(n - 2, n - 1, k, n)];
          v += 2 * (*curr)[TENSOR_IDX(n - 1, n - 2, k, n)];
          v += (*curr)[TENSOR_IDX(n - 1, n - 1, k - 1, n)];
          v += (*curr)[TENSOR_IDX(n - 1, n - 1, k + 1, n)];
          v -= delta * delta * source[TENSOR_IDX(n - 1, n - 1, k, n)];
          v /= 6;
          (*next)[TENSOR_IDX(n - 1, n - 1, k, n)] = v;
        }
      }
// Outer Vertices
#pragma omp section
      {
        // (0, 0, 0)
        v = 0;
        v += 2 * (*curr)[TENSOR_IDX(0, 0, 1, n)];
        v += 2 * (*curr)[TENSOR_IDX(0, 1, 0, n)];
        v += 2 * (*curr)[TENSOR_IDX(1, 0, 0, n)];
        v -= delta * delta * source[TENSOR_IDX(0, 0, 0, n)];
        v /= 6;
        (*next)[TENSOR_IDX(0, 0, 0, n)] = v;
      }
#pragma omp section
      {
        // (0, 0, n-1)
        v = 0;
        v += 2 * (*curr)[TENSOR_IDX(0, 0, n - 2, n)];
        v += 2 * (*curr)[TENSOR_IDX(0, 1, n - 1, n)];
        v += 2 * (*curr)[TENSOR_IDX(1, 0, n - 1, n)];
        v -= delta * delta * source[TENSOR_IDX(0, 0, n - 1, n)];
        v /= 6;
        (*next)[TENSOR_IDX(0, 0, n - 1, n)] = v;
      }
#pragma omp section
      {
        // (0, n-1, 0)
        v = 0;
        v += 2 * (*curr)[TENSOR_IDX(0, n - 2, 0, n)];
        v += 2 * (*curr)[TENSOR_IDX(0, n - 1, 1, n)];
        v += 2 * (*curr)[TENSOR_IDX(1, n - 1, 0, n)];
        v -= delta * delta * source[TENSOR_IDX(0, n - 1, 0, n)];
        v /= 6;
        (*next)[TENSOR_IDX(0, n - 1, 0, n)] = v;
      }
#pragma omp section
      {
        // (0, n-1, n-1)
        v = 0;
        v += 2 * (*curr)[TENSOR_IDX(0, n - 2, n - 1, n)];
        v += 2 * (*curr)[TENSOR_IDX(0, n - 1, n - 2, n)];
        v += 2 * (*curr)[TENSOR_IDX(1, n - 1, n - 1, n)];
        v -= delta * delta * source[TENSOR_IDX(0, n - 1, n - 1, n)];
        v /= 6;
        (*next)[TENSOR_IDX(0, n - 1, n - 1, n)] = v;
      }
#pragma omp section
      {
        // (n-1, 0, 0)
        v = 0;
        v += 2 * (*curr)[TENSOR_IDX(n - 2, 0, 0, n)];
        v += 2 * (*curr)[TENSOR_IDX(n - 1, 0, 1, n)];
        v += 2 * (*curr)[TENSOR_IDX(n - 1, 1, 0, n)];
        v -= delta * delta * source[TENSOR_IDX(n - 1, 0, 0, n)];
        v /= 6;
        (*next)[TENSOR_IDX(n - 1, 0, 0, n)] = v;
      }
#pragma omp section
      {
        // (n-1, 0, n-1)
        v = 0;
        v += 2 * (*curr)[TENSOR_IDX(n - 2, 0, n - 1, n)];
        v += 2 * (*curr)[TENSOR_IDX(n - 1, 0, n - 2, n)];
        v += 2 * (*curr)[TENSOR_IDX(n - 1, 1, n - 1, n)];
        v -= delta * delta * source[TENSOR_IDX(n - 1, 0, n - 1, n)];
        v /= 6;
        (*next)[TENSOR_IDX(n - 1, 0, n - 1, n)] = v;
      }
#pragma omp section
      {
        // (n-1, n-1, 0)
        v = 0;
        v += 2 * (*curr)[TENSOR_IDX(n - 2, n - 1, 0, n)];
        v += 2 * (*curr)[TENSOR_IDX(n - 1, n - 2, 0, n)];
        v += 2 * (*curr)[TENSOR_IDX(n - 1, n - 1, 1, n)];
        v -= delta * delta * source[TENSOR_IDX(n - 1, n - 1, 0, n)];
        v /= 6;
        (*next)[TENSOR_IDX(n - 1, n - 1, 0, n)] = v;
      }
#pragma omp section
      {
        // (n-1, n-1, n-1)
        v = 0;
        v += 2 * (*curr)[TENSOR_IDX(n - 2, n - 1, n - 1, n)];
        v += 2 * (*curr)[TENSOR_IDX(n - 1, n - 2, n - 1, n)];
        v += 2 * (*curr)[TENSOR_IDX(n - 1, n - 1, n - 2, n)];
        v -= delta * delta * source[TENSOR_IDX(n - 1, n - 1, n - 1, n)];
        v /= 6;
        (*next)[TENSOR_IDX(n - 1, n - 1, n - 1, n)] = v;
      }
    }
#pragma omp for
    // Inner Cube
    for (i = 1; i < n - 1; i++)
    {
      for (j = 1; j < n - 1; j++)
      {
        for (k = 1; k < n - 1; k++)
        {
          v = 0;
          v += (*curr)[TENSOR_IDX(i - 1, j, k, n)];
          v += (*curr)[TENSOR_IDX(i, j - 1, k, n)];
          v += (*curr)[TENSOR_IDX(i, j, k - 1, n)];
          v += (*curr)[TENSOR_IDX(i, j, k + 1, n)];
          v += (*curr)[TENSOR_IDX(i, j + 1, k, n)];
          v += (*curr)[TENSOR_IDX(i + 1, j, k, n)];
          v -= delta * delta * source[TENSOR_IDX(i, j, k, n)];
          v /= 6;
          (*next)[TENSOR_IDX(i, j, k, n)] = v;
        }
      }
    }
#pragma omp single
    {
      std::vector<float> *temp = curr;
      curr = next;
      next = temp;
    }
  }
  if (debug)
  {
    auto time_stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time_stop - time_start);
    std::cout << "Duration: " << duration.count() << std::endl;
  }

  delete next;
  return curr;
}
