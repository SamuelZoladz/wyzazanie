#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#include <cmath>
#include <cstdint>
#include <vector>

double calc_quadratic_function(const std::vector<double>& x, const uint32_t n);
std::vector<double> make_quadratic_x0(uint32_t n);
double l2_norm(const std::vector<double>& x);


double calc_woods_function( const std::vector<double>& x, const uint32_t n);
std::vector<double> make_woods_x0(uint32_t n);
double l2_norm_distance_to_woods_min(const std::vector<double>& x, uint32_t n);

double calc_powell_singular_function(const std::vector<double>& x, uint32_t n);
std::vector<double> make_powell_x0(uint32_t n);
double l2_norm_distance_to_powell_min(const std::vector<double>& x, uint32_t n);

#endif //FUNCTIONS_H
