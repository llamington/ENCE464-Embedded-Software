#ifndef __POISSON_SOLVER_H
#define __POISSON_SOLVER_H

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