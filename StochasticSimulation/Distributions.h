#pragma once

#include <string>
enum class Distributions
{
	RECTANGLE, NORM, PARABOL, TRIANGLE, LOCALIZE, NORM1D
};

std::string distrib_name(Distributions distr);
double init_func(double x, Distributions distr, double a, double b, double c);
double gaussFunc(double a, double sigma, double x);

double multivariateStandardNormal(double x1, double x2, double x3);
