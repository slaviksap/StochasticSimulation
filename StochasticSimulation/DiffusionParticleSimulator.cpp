#include "DiffusionParticleSimulator.h"
#include <cmath>
#include <omp.h>

using namespace boost::qvm;
const double Pi = 3.1415926535897;
void DiffusionParticleSimulator::Calculate()
{
	double startEnergy = 0, endEnergy = 0, startImpuls = 0, endImpuls = 0;
	layer.assign(N, Vec3{ 0,0,0 });
	double a_mean = 0;
	Mat3 sigma_mean = Mat3{ 0,0,0,0,0,0,0,0,0 };
	for (int m = 1; m <= M; ++m)
	{
		initialStochasticDistribution();
		prev_layer = first_layer;
		double t = 0;

		for (int i = 1; i <= k; ++i)
		{
			t += dt;
			//layer.assign(N, Vec3{ 0,0,0 });
			for (int j = 0; j < N; ++j)
			{
				Vec3 v = prev_layer[j];
				auto a = demolition(v, t);
				auto sigma = volatility(v, t);
				a_mean += a;
				sigma_mean += sigma;
				layer[j] = (v -  v * a * dt + sigma * Randomizer::sampleNormVec3() * sdt);
			}
			if (i != k)
				prev_layer = layer;
		}
		startEnergy += Energy(first_layer);
		startImpuls += Impuls(first_layer);
		endEnergy += Energy(layer);
		endImpuls += Impuls(layer);
		replenish_histogram(init_result, first_layer);
		replenish_histogram(result, layer);
		cout << m << endl;
	}
	for (auto& i : init_result)
		i /= M;
	for (auto& i : result)
		i /= M;
	startEnergy /= M;
	startImpuls /= M;
	endEnergy /= M;
	endImpuls /= M;
	a_mean /= k * N * M;
	sigma_mean /= k * N * M;
	std::cout << "Start energy = " << startEnergy << endl << "Start impuls = " << startImpuls << endl;
	std::cout << "End energy = " << endEnergy << endl << "End Impuls = " << endImpuls << endl;
	std::cout << "Energy change = " << startEnergy / endEnergy << endl << "Impuls change = " << startImpuls / endImpuls << endl;
	std::cout << "a mean = " << a_mean << endl;
	std::cout << "sigma mean = \n" << sigma_mean.a[0][0] << "\t" << sigma_mean.a[0][1] << "\t" << sigma_mean.a[0][2] << endl <<
		sigma_mean.a[1][0] << "\t" << sigma_mean.a[1][1] << "\t" << sigma_mean.a[1][2] << endl <<
		sigma_mean.a[2][0] << "\t" << sigma_mean.a[2][1] << "\t" << sigma_mean.a[2][2] << endl;
}

double DiffusionParticleSimulator::demolition(const Vec3& v, double t)
{
	double c = mag(v);
	if (c == 0.0)
		return 0;
	double right = sqrt(Pi) * erf(c) * (2 * c * c + 2 - 1 / (2 * c * c)) + exp(-c * c) * (2 * c + 1 / c);
	double coef = sqrt(Pi) * right / (4 * c);
	return coef;
}

Mat3 DiffusionParticleSimulator::volatility(const Vec3& v, double t)
{
	const double c = mag(v);
	if (c == 0.0)
		return zero_mat<double, 3, 3>();
	const double perf = sqrt(Pi) * erf(c);
	const double c2 = c * c;
	const double c3 = c2 * c;
	const double P = perf * (c3 / 3 + 1.5 * c + 0.75 / c - 0.125 / c3) + exp(-c2) * (c2 / 3 + 4.0 / 3 + 0.25 / c2);
	const double S = perf * (c + 1.5 / c - 0.75 / c3 + 3.0 / (8 * pow(c, 5))) + exp(-c2) * (1 + 1 / c2 - 3.0 / (4 * pow(c, 4)));
	const double lambda1 = c2 * S + P;

	if (P < 0 || lambda1 < 0)
		return Mat3{ 0,0,0,0,0,0,0,0,0 };
	Mat3 mat;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
		{
			if (i == j)
			{
				mat.a[i][j] = pow(Pi, 0.25) / 2 * (sqrt(P) + v.a[i] * v.a[j] / c2 * (sqrt(lambda1) - sqrt(P)));
			}
			else
				mat.a[i][j] = pow(Pi, 0.25) / 2 * (v.a[i] * v.a[j] / c2 * (sqrt(lambda1) - sqrt(P)));

		}	
	//Mat3 mat{pow(Pi, 0.25) / 2 * sqrt(P + v.a[0] * v.a[0] * S),0,0,0,0,0,0,0,0 };
	return mat;
}
