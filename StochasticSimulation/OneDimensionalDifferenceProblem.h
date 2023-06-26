#pragma once
#include"Distributions.h"
#include<vector>
#include<random>
#include<fstream>

class BolzmanDiff
{
public:
	std::vector<std::vector<double>> net;
	double t = 1;
	double dt, dx;
	double a, b;
	int N, k;
	BolzmanDiff(double a, double b, double t, int N, int k) : a(a), b(b), t(t), N(N), k(k), net(k, std::vector<double>(N,0))
	{
		dt = t / k;
		dx = (b - a) / N;
	}
	void InitialConditions(double a, double sigma)
	{
		for (int i = 0; i < N; ++i)
			net[0][i] = gaussFunc(a, sigma, this->a + dx * i);
	}
	void Calculate()
	{
		InitialConditions(0, 1);
		std::vector<double> a_coef(N);
		std::vector<double> sigma_coef(N);
		for (int i = 0; i < N; ++i)
		{
			a_coef[i] = demolition(a + i * dx);
			sigma_coef[i] = sigmasqr(abs(a + i * dx));
		}
		for (int n = 0; n < k - 1; ++n)
		{
			for (int i = 1; i < N - 1; ++i)
				net[n + 1][i] = net[n][i] - (dt / (2 * dx)) * (a_coef[i + 1] * net[n][i + 1] - a_coef[i - 1] * net[n][i - 1])
				+ (dt / (2 * dx * dx)) * (sigma_coef[i + 1] * net[n][i + 1] - 2 * sigma_coef[i] * net[n][i] + sigma_coef[i - 1] * net[n][i - 1]);
			if (n % 1000 == 0)
				cout << n << endl;
		}
		std::ofstream file;
		file.open("Tables\\init_difference.txt");
		for (int i = 0; i < N; ++i)
			file << a + dx * i << "\t" << net[0][i] << "\n";
		file.close();
		file.open("Tables\\last_difference.txt");
		for (int i = 0; i < N; ++i)
			file << a + dx * i << "\t" << net[k - 1][i] << "\n";
		file.close();
	}

	double demolition(double v)
	{
		return 0;
		double c = abs(v);
		if (c < 0.01)
			return 2.3;
		double right = sqrt(std::_Pi) * erf(c) * (2 * c * c + 2 - 1 / (2 * c * c)) + exp(-c * c) * (2 * c + 1 / c);
		double coef = v * sqrt(std::_Pi) * right / (4 * c);
		return coef;
	}
	double sigmasqr(double c)
	{
		if (c < 0.01)
			return 2.56;
		const double perf = sqrt(std::_Pi) * erf(c);
		const double c2 = c * c;
		const double c3 = c2 * c;
		const double P = perf * (c3 / 3 + 1.5 * c + 0.75 / c - 0.125 / c3) + exp(-c2) * (c2 / 3 + 4.0 / 3 + 0.25 / c2);
		const double S = perf * (c + 1.5 / c - 0.75 / c3 + 3.0 / (8 * pow(c, 5))) + exp(-c2) * (1 + 1 / c2 - 3.0 / (4 * pow(c, 4)));
		if (P + c2 * S < 0)
			return 0;
		return sqrt(std::_Pi) / 4 * (P + c2 * S);
	}
};