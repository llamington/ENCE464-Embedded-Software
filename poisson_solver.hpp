#ifndef __POISSON_SOLVER_H
#define __POISSON_SOLVER_H

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>

class PoissonSolver
{
private:
    int n;
    const std::vector<double> &source;
    int iterations;
    float delta;
    int threads;
    std::vector<double> *curr;
    std::vector<double> *next;
    bool debug = false;
    int block_size = (n + threads - 1) / threads;

    // Mutex to coordinate read/writes to current
    std::mutex curr_mut;

    // Counts how many threads have completed an iteration
    int threads_waiting = 0;
    std::condition_variable barrier;
    // Tracks whether the original current pointer is active
    bool original_curr = true;

    void poisson_thread(int thread_num);

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