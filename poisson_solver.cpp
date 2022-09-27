#include "poisson_solver.hpp"
#include "util.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

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
      curr(new std::vector<double>),
      next(new std::vector<double>) {}

void PoissonSolver::poisson_thread(int thread_num)
{
  std::vector<double> *temp;
  double v = 0;

  for (int iter = 0; iter < iterations; iter++)
  {
    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < n; j++)
      {
        for (int k = 0; k < n; k++)
        {
          v = 0;

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
    temp = curr;
    curr = next;
    next = temp;
  }

  pthread_exit(NULL);
}

std::vector<double> *PoissonSolver::solve(void)
{
  auto time_start = std::chrono::high_resolution_clock::now();
  try
  {
    curr->reserve(n * n * n);
    next->reserve(n * n * n);
  }
  catch (std::bad_alloc &)
  {
    std::cerr << "Error: ran out of memory when trying to allocate " << n << " sized cube" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::vector<std::thread> threads_vec;
  threads_vec.reserve(threads);

  for (int i = 0; i < threads; ++i)
  {
    threads_vec.push_back(std::thread(&PoissonSolver::poisson_thread, this, i));
  }
  for (auto &thread : threads_vec)
    thread.join();

  auto time_stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time_stop - time_start);
  std::cout << "Duration: " << duration.count() << std::endl;

  return curr;
}