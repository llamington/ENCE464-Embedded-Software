#ifndef __POISSON_SOLVER_H
#define __POISSON_SOLVER_H

#include <vector>

class PoissonSolver
{
private:
    const int n;
    const std::vector<float> &source;
    const int iterations;
    const int threads;
    const float delta;
    bool debug = false;
    std::vector<float> *curr;
    std::vector<float> *next;

public:
    PoissonSolver(int n,
                  const std::vector<float> &source,
                  int iterations,
                  int threads,
                  float delta,
                  bool debug);

    std::vector<float> *solve(void);
};

#endif