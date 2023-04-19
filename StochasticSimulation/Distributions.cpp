#include "Distributions.h"
#include <cmath>

std::string distrib_name(Distributions distr)
{
	if (distr == Distributions::RECTANGLE)
		return "Rectangle";
	if (distr == Distributions::NORM)
		return "Gauss";
	if (distr == Distributions::PARABOL)
		return "Parabolic";
	if (distr == Distributions::TRIANGLE)
		return "Triangle";
	if (distr == Distributions::LOCALIZE)
		return "Localize";
}
double init_func(double x, Distributions distr, double a, double b, double c)
{
	if (distr == Distributions::RECTANGLE)
	{
		if (x >= a && x <= b)
			return c;
		return 0;
	}
	if (distr == Distributions::PARABOL)
	{
		double f = a * x * x + b * x + c;
		if (f > 0)
			return f;
		return 0;
	}
	if (distr == Distributions::NORM)
	{
		return c / (b * 4.442882) * exp(-(x - a) * (x - a) / (2 * b * b));
	}
	if (distr == Distributions::TRIANGLE)
	{
		if (x >= a && x <= c)
			return (x - a) / (c - a);
		if (x > c && x <= b)
			return (b - x) / (b - c);
		return 0;
	}
	if (distr == Distributions::LOCALIZE)
	{
		double Pi = 3.14159265;
		double k0 = 0.2;
		double q0 = 4;
		double Lt = 2 * Pi * sqrt(k0 / q0 * (a + 1) / (a * a));
		if (abs(x) > Lt / 2)
			return 0;
		double trig = cos(Pi * x / Lt);
		double temp = 2 * (a + 1) / (a * (a + 2)) * trig * trig;
		return pow(temp, 1 / a) / pow(q0 * b, 1 / a);
	}
}