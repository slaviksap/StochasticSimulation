#pragma once

#include <string>
enum class Distributions
{
	RECTANGLE, NORM, PARABOL, TRIANGLE, LOCALIZE
};

std::string distrib_name(Distributions distr);
double init_func(double x, Distributions distr, double a, double b, double c);