#include "poisson_solver.hpp"
#include "util.hpp"
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

#define DELTA 1.0

int main(int argc, char *argv[])
{
    int threads = omp_get_max_threads() - 3;
    int n = 7;
    int iterations = 300;
    bool debug = false;
    std::string temp_arg;

    // parse the command line arguments
    for (int i = 1; i < argc; ++i)
    {
        temp_arg = std::string(argv[i]);

        if (temp_arg == "-h" || temp_arg == "--help")
        {
            std::cout << "Usage: poisson [-n size] [-i iterations] [-t threads] [--debug]" << std::endl;
            return EXIT_SUCCESS;
        }

        if (temp_arg == "-n")
        {
            if (i == argc - 1)
            {
                std::cerr << "Error: expected size after -n!" << std::endl;
                return EXIT_FAILURE;
            }
            n = std::atoi(argv[++i]);
        }

        if (temp_arg == "-i")
        {
            if (i == argc - 1)
            {
                std::cerr << "Error: expected iterations after -i!" << std::endl;
                return EXIT_FAILURE;
            }
            iterations = std::atoi(argv[++i]);
        }

        if (temp_arg == "-t")
        {
            if (i == argc - 1)
            {
                std::cerr << "Error: expected threads after -t!" << std::endl;
                return EXIT_FAILURE;
            }
            threads = std::atoi(argv[++i]);
        }

        if (temp_arg == "--debug")
            debug = true;
    }

    // Ensure we have an odd sized cube
    if (n % 2 == 0)
    {
        std::cerr << "Error: n should be an odd number!" << std::endl;
        return EXIT_FAILURE;
    }

    // Create a source term with a single point in the centre
    std::vector<float> source;
    try
    {
        source = std::vector<float>(n * n * n, 0);
    }
    catch (std::bad_alloc &)
    {
        std::cerr << "Error: failed to allocated source term (n=" << n << ")" << std::endl;
        return EXIT_FAILURE;
    }

    source[TENSOR_IDX(n / 2, n / 2, n / 2, n)] = 1;

    PoissonSolver poisson_solver(n, source, iterations, threads, DELTA, debug);
    // Calculate the resulting field with Neumann conditions
    auto result = poisson_solver.solve();

    std::cout << std::fixed << std::setprecision(5);

    // Print out the middle slice of the cube for validation
    for (int x = 0; x < n; ++x)
    {
        for (int y = 0; y < n; ++y)
            std::cout << (*result)[TENSOR_IDX(n / 2, y, x, n)] << " ";

        std::cout << std::endl;
    }
    return EXIT_SUCCESS;
}
