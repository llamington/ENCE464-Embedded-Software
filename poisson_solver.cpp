#include "poisson_solver.hpp"
#include "util.hpp"
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
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
  double v = 0;
  int start_i = thread_num * block_size;
  int end_i = (thread_num + 1) * block_size;
  end_i = (end_i > n) ? n : end_i;

  for (int iter = 0; iter < iterations; iter++)
  {
    for (int i = start_i; i < end_i; i++)
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
    {
      std::unique_lock<std::mutex> lock(curr_mut);
      // Initialise barrier condition
      bool original_curr_it = original_curr;

      // Notify if barrier is met
      if (++threads_waiting == threads)
      {
        threads_waiting = 0;
        std::vector<double> *temp = curr;
        curr = next;
        next = temp;
        original_curr = !original_curr;
        barrier.notify_all();
      }
      else
        barrier.wait(lock, [&original_curr_it, this]()
                     { return original_curr_it != original_curr; });
    }
  }
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
  // for (int i = 0; i < iterations; ++i)
  // {
  // }
  for (auto &thread : threads_vec)
    thread.join();

  auto time_stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time_stop - time_start);
  std::cout << "Duration: " << duration.count() << std::endl;

  delete next;
  return curr;
}