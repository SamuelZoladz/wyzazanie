#include "functions.h"
#include <omp.h>

// =======================================================
// QUADRATIC FUNCTION
// =======================================================

double calc_quadratic_function(const std::vector<double>& x, const uint32_t n)
{
    double res = 0.0;

    #pragma omp parallel for reduction(+:res)
    for (uint32_t i = 3; i < n; ++i)
    {
        res += 100.0 * (x[i] * x[i] + x[i - 1] * x[i - 1])
             +        x[i - 2] * x[i - 2];
    }
    return res;
}

std::vector<double> make_quadratic_x0(uint32_t n)
{
    return std::vector<double>(n, 3.0);
}

double l2_norm(const std::vector<double>& x)
{
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < x.size(); ++i)
    {
        sum += x[i] * x[i];
    }
    return std::sqrt(sum);
}

// =======================================================
// WOODS FUNCTION
// =======================================================

double calc_woods_function(const std::vector<double>& x, const uint32_t n)
{
    double res = 0.0;
    const uint32_t blocks = n / 4;

    #pragma omp parallel for reduction(+:res)
    for (uint32_t i = 0; i < blocks; ++i)
    {
        const uint32_t idx = 4 * i;

        const double x1 = x[idx];
        const double x2 = x[idx + 1];
        const double x3 = x[idx + 2];
        const double x4 = x[idx + 3];

        const double t1 = x2 - x1 * x1;
        const double t2 = 1.0 - x1;
        const double t3 = x4 - x3 * x3;
        const double t4 = 1.0 - x3;
        const double t5 = x2 + x4 - 2.0;
        const double t6 = x2 - x4;

        res += 100.0 * t1 * t1
             +        t2 * t2
             + 90.0 * t3 * t3
             +        t4 * t4
             + 10.0 * t5 * t5
             +  0.1 * t6 * t6;
    }
    return res;
}

std::vector<double> make_woods_x0(uint32_t n)
{
    std::vector<double> x0(n);

    for (uint32_t i = 0; i < n; i += 4)
    {
        x0[i]     = -3.0;
        x0[i + 1] = -1.0;
        x0[i + 2] = -3.0;
        x0[i + 3] = -1.0;
    }
    return x0;
}

double l2_norm_distance_to_woods_min(const std::vector<double>& x, uint32_t n)
{
    double sum = 0.0;
    const uint32_t blocks = n / 4;

    #pragma omp parallel for reduction(+:sum)
    for (uint32_t i = 0; i < blocks; ++i)
    {
        const uint32_t idx = 4 * i;

        const double dx = x[idx]     - 1.0;
        const double dy = x[idx + 1] - 1.0;
        const double dz = x[idx + 2] - 1.0;
        const double dw = x[idx + 3] - 1.0;

        sum += dx * dx + dy * dy + dz * dz + dw * dw;
    }
    return std::sqrt(sum);
}

// =======================================================
// POWELL SINGULAR FUNCTION
// =======================================================

double calc_powell_singular_function(const std::vector<double>& x, uint32_t n)
{
    double res = 0.0;
    const uint32_t blocks = n / 4;

    #pragma omp parallel for reduction(+:res)
    for (uint32_t i = 0; i < blocks; ++i)
    {
        const uint32_t idx = 4 * i;

        const double x1 = x[idx];
        const double x2 = x[idx + 1];
        const double x3 = x[idx + 2];
        const double x4 = x[idx + 3];

        const double t1 = x1 + 10.0 * x2;
        const double t2 = x3 - x4;
        const double t3 = x2 - 2.0 * x3;
        const double t4 = x1 - x4;

        res +=        t1 * t1
             +  5.0 * t2 * t2
             +        t3 * t3 * t3 * t3
             + 10.0 * t4 * t4 * t4 * t4;
    }
    return res;
}

std::vector<double> make_powell_x0(uint32_t n)
{
    std::vector<double> x0(n);

    for (uint32_t i = 0; i < n; i += 4)
    {
        x0[i]     = 3.0;
        x0[i + 1] = -1.0;
        x0[i + 2] = 0.0;
        x0[i + 3] = 1.0;
    }
    return x0;
}

double l2_norm_distance_to_powell_min(const std::vector<double>& x, uint32_t n)
{
    double sum = 0.0;
    const uint32_t blocks = n / 4;

    #pragma omp parallel for reduction(+:sum)
    for (uint32_t i = 0; i < blocks; ++i)
    {
        const uint32_t idx = 4 * i;

        sum += x[idx]     * x[idx]
             + x[idx + 1] * x[idx + 1]
             + x[idx + 2] * x[idx + 2]
             + x[idx + 3] * x[idx + 3];
    }
    return std::sqrt(sum);
}
