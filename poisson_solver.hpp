#ifndef __POISSON_SOLVER_H
#define __POISSON_SOLVER_H

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