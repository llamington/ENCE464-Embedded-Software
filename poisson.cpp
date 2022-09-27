#include "poisson_solver.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

/**
 * poisson.c
 * Implementation of a Poisson solver with Neumann boundary conditions.
 *
 * This template handles the basic program launch, argument parsing, and memory
 * allocation required to implement the solver *at its most basic level*. You
 * will likely need to allocate more memory, add threading support, account for
 * cache locality, etc...
 *
 * BUILDING:
 * gcc -o poisson poisson.c -lpthread
 *
 * [note: linking pthread isn't strictly needed until you add your
 *        multithreading code]
 *
 * TODO:
 * 1 - Read through this example, understand what it does and what it gives you
 *     to work with.
 * 2 - Implement the basic algorithm and get a correct output.
 * 3 - Add a timer to track how long your execution takes.
 * 4 - Profile your solution and identify weaknesses.
 * 5 - Improve it!
 * 6 - Remember that this is now *your* code and *you* should modify it however
 *     needed to solve the assignment.
 *
 * See the lab notes for a guide on profiling and an introduction to
 * multithreading (see also threads.c which is reference by the lab notes).
 */

// Global flag
// Set to true when operating in debug mode to enable verbose logging
static bool debug = false;

int main(int argc, char *argv[])
{
    int threads = 1;
    int n = 5;
    float delta = 1;
    int iterations = 10;

    // parse the command line arguments
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            std::cout << "Usage: poisson [-n size] [-i iterations] [-t threads] [--debug]" << std::endl;
            return EXIT_SUCCESS;
        }

        if (strcmp(argv[i], "-n") == 0)
        {
            if (i == argc - 1)
            {
                std::cerr << "Error: expected size after -n!" << std::endl;
                return EXIT_FAILURE;
            }

            n = atoi(argv[++i]);
        }

        if (strcmp(argv[i], "-i") == 0)
        {
            if (i == argc - 1)
            {
                std::cerr << "Error: expected iterations after -i!" << std::endl;
                return EXIT_FAILURE;
            }

            iterations = atoi(argv[++i]);
        }

        if (strcmp(argv[i], "-t") == 0)
        {
            if (i == argc - 1)
            {
                std::cerr << "Error: expected threads after -t!" << std::endl;
                return EXIT_FAILURE;
            }

            threads = atoi(argv[++i]);
        }

        if (strcmp(argv[i], "--debug") == 0)
        {
            debug = true;
        }
    }

    // Ensure we have an odd sized cube
    if (n % 2 == 0)
    {
        std::cerr << "Error: n should be an odd number!" << std::endl;
        return EXIT_FAILURE;
    }

    // Create a source term with a single point in the centre
    std::vector<double> source;
    try
    {
        source.reserve(n * n * n);
    }
    catch (std::bad_alloc &)
    {
        std::cerr << "Error: failed to allocated source term (n=" << n << ")" << std::endl;
        return EXIT_FAILURE;
    }

    source[(n * n * n) / 2] = 1;

    PoissonSolver poisson_solver(n, source, iterations, threads, delta, debug);
    // Calculate the resulting field with Neumann conditions
    auto result = poisson_solver.solve();

    std::cout << "Result:" << std::endl;
    // Print out the middle slice of the cube for validation
    for (int x = 0; x < n; ++x)
    {
        for (int y = 0; y < n; ++y)
        {
            printf("%0.5f ", (*result)[((n / 2) * n + y) * n + x]);
        }
        std::cout << std::endl;
    }
    return EXIT_SUCCESS;
}
