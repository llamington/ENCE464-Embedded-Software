#include "poisson_solver.hpp"
#include "util.hpp"
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <thread>

PoissonSolver::PoissonSolver(std::size_t n,
                             const std::vector<double> &source,
                             int iterations,
                             int threads,
                             float delta,
                             bool debug)
    : n(n),
      source(source),
      iterations(iterations),
      delta(delta),
      threads(threads)
{
  try
  {
    curr = new std::vector<double>(n * n * n);
    next = new std::vector<double>(n * n * n);
  }
  catch (std::bad_alloc &)
  {
    std::cerr << "Error: ran out of memory when trying to allocate " << n << " sized cube" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void PoissonSolver::poisson_thread(int thread_num)
{
  std::unique_lock<std::mutex> lock(curr_mut, std::defer_lock);
  double v = 0;
  std::size_t start_i = (thread_num - 1) * block_size;
  start_i = (start_i < 1) ? 1 : start_i;

  std::size_t end_i = thread_num * block_size;
  end_i = (end_i > n - 1) ? n - 1 : end_i;
  std::size_t i, j, k;

  for (uint16_t iter = 0; iter < iterations; iter++)
  {
    // auto time_start = std::chrono::high_resolution_clock::now();

    if (thread_num == 0)
    {
      // Outer Faces
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
      {
        // (0, 0, 0)
        {
          v = 0;
          v += 2 * (*curr)[TENSOR_IDX(0, 0, 1, n)];
          v += 2 * (*curr)[TENSOR_IDX(0, 1, 0, n)];
          v += 2 * (*curr)[TENSOR_IDX(1, 0, 0, n)];
          v -= delta * delta * source[TENSOR_IDX(0, 0, 0, n)];
          v /= 6;
          (*next)[TENSOR_IDX(0, 0, 0, n)] = v;
        }

        // (0, 0, n-1)
        {
          v = 0;
          v += 2 * (*curr)[TENSOR_IDX(0, 0, n - 2, n)];
          v += 2 * (*curr)[TENSOR_IDX(0, 1, n - 1, n)];
          v += 2 * (*curr)[TENSOR_IDX(1, 0, n - 1, n)];
          v -= delta * delta * source[TENSOR_IDX(0, 0, n - 1, n)];
          v /= 6;
          (*next)[TENSOR_IDX(0, 0, n - 1, n)] = v;
        }

        // (0, n-1, 0)
        {
          v = 0;
          v += 2 * (*curr)[TENSOR_IDX(0, n - 2, 0, n)];
          v += 2 * (*curr)[TENSOR_IDX(0, n - 1, 1, n)];
          v += 2 * (*curr)[TENSOR_IDX(1, n - 1, 0, n)];
          v -= delta * delta * source[TENSOR_IDX(0, n - 1, 0, n)];
          v /= 6;
          (*next)[TENSOR_IDX(0, n - 1, 0, n)] = v;
        }

        // (0, n-1, n-1)
        {
          v = 0;
          v += 2 * (*curr)[TENSOR_IDX(0, n - 2, n - 1, n)];
          v += 2 * (*curr)[TENSOR_IDX(0, n - 1, n - 2, n)];
          v += 2 * (*curr)[TENSOR_IDX(1, n - 1, n - 1, n)];
          v -= delta * delta * source[TENSOR_IDX(0, n - 1, n - 1, n)];
          v /= 6;
          (*next)[TENSOR_IDX(0, n - 1, n - 1, n)] = v;
        }

        // (n-1, 0, 0)
        {
          v = 0;
          v += 2 * (*curr)[TENSOR_IDX(n - 2, 0, 0, n)];
          v += 2 * (*curr)[TENSOR_IDX(n - 1, 0, 1, n)];
          v += 2 * (*curr)[TENSOR_IDX(n - 1, 1, 0, n)];
          v -= delta * delta * source[TENSOR_IDX(n - 1, 0, 0, n)];
          v /= 6;
          (*next)[TENSOR_IDX(n - 1, 0, 0, n)] = v;
        }

        // (n-1, 0, n-1)
        {
          v = 0;
          v += 2 * (*curr)[TENSOR_IDX(n - 2, 0, n - 1, n)];
          v += 2 * (*curr)[TENSOR_IDX(n - 1, 0, n - 2, n)];
          v += 2 * (*curr)[TENSOR_IDX(n - 1, 1, n - 1, n)];
          v -= delta * delta * source[TENSOR_IDX(n - 1, 0, n - 1, n)];
          v /= 6;
          (*next)[TENSOR_IDX(n - 1, 0, n - 1, n)] = v;
        }

        // (n-1, n-1, 0)
        {
          v = 0;
          v += 2 * (*curr)[TENSOR_IDX(n - 2, n - 1, 0, n)];
          v += 2 * (*curr)[TENSOR_IDX(n - 1, n - 2, 0, n)];
          v += 2 * (*curr)[TENSOR_IDX(n - 1, n - 1, 1, n)];
          v -= delta * delta * source[TENSOR_IDX(n - 1, n - 1, 0, n)];
          v /= 6;
          (*next)[TENSOR_IDX(n - 1, n - 1, 0, n)] = v;
        }

        // (n-1, n-1, n-1)
        {
          v = 0;
          v += 2 * (*curr)[TENSOR_IDX(n - 2, n - 1, n - 1, n)];
          v += 2 * (*curr)[TENSOR_IDX(n - 1, n - 2, n - 1, n)];
          v += 2 * (*curr)[TENSOR_IDX(n - 1, n - 1, n - 2, n)];
          v -= delta * delta * source[TENSOR_IDX(n - 1, n - 1, n - 1, n)];
          v /= 6;
          (*next)[TENSOR_IDX(n - 1, n - 1, n - 1, n)] = v;
        }
      }
    }

    // Inner Cube
    for (i = start_i; i < end_i; i++)
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

    // auto time_stop = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time_stop - time_start);

    // Initialise barrier condition
    bool original_curr_it = original_curr;

    lock.lock();
    // std::cout << "Thread " << thread_num << ": " << duration.count() << std::endl;

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
    lock.unlock();
  }
}

std::vector<double> *PoissonSolver::solve(void)
{
  auto time_start = std::chrono::high_resolution_clock::now();

  std::vector<std::thread> threads_vec(threads);

  for (int i = 0; i < threads; ++i)
  {
    threads_vec[i] = std::thread(&PoissonSolver::poisson_thread, this, i);
  }

  for (auto &thread : threads_vec)
    thread.join();

  auto time_stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time_stop - time_start);
  std::cout << "Duration: " << duration.count() << std::endl;

  delete next;
  return curr;
}