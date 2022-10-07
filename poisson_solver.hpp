#ifndef __POISSON_SOLVER_H
#define __POISSON_SOLVER_H

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>

class PoissonSolver
{
private:
    const int n;
    const std::vector<double> &source;
    const int iterations;
    const int threads;
    const float delta;
    bool debug = false;
    std::vector<double> *curr;
    std::vector<double> *next;
    // const int block_size = (n + threads - 4) / (threads - 1);

    // Mutex to coordinate read/writes to current
    // std::mutex curr_mut;

    // Counts how many threads have completed an iteration
    // int threads_waiting = 0;
    // std::condition_variable barrier;
    // Tracks whether the original current pointer is active
    // bool original_curr = true;

    // void poisson_thread(int thread_num);

public:
    PoissonSolver(int n,
                  const std::vector<double> &source,
                  int iterations,
                  int threads,
                  float delta,
                  bool debug);

    std::vector<double> *solve(void);
};

#endif