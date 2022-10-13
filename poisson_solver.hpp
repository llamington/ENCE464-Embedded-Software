#ifndef __POISSON_SOLVER_H
#define __POISSON_SOLVER_H

#include <vector>

/**
 * @brief Class for solving Poisson's equation using Jacobi Relaxation
 */
class PoissonSolver
{
private:
    const int n;
    const std::vector<float> &source;
    const int iterations;
    const int threads;
    const float delta;
    const bool debug;
    std::vector<float> *curr;
    std::vector<float> *next;

public:
    /**
     * @brief Construct a new Poisson Solver object
     *
     * @param n: Size of cube (nxnxn)
     * @param source:  Source field volume
     * @param iterations: Number of iterations of the Jacobi Relaxation
     * @param threads: Number of threads to perform computation on
     * @param delta: Distance between volume voxels
     * @param debug: Whether debugging data is printed to stdout
     */
    PoissonSolver(int n,
                  const std::vector<float> &source,
                  int iterations,
                  int threads,
                  float delta,
                  bool debug);

    /**
     * @brief Solves Poisson's equation using Jacobi Relaxation
     * @return Field after the specified number of iterations
     */
    std::vector<float> *solve(void);
};

#endif