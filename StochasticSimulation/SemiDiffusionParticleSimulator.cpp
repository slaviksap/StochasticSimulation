#include "SemiDiffusionParticleSimulator.h"
#include <armadillo>
using namespace boost;
const double Pi = 3.1415926535897;
void SemiDiffusionParticleSimulator::Calculate()
{
	double startEnergy = 0, endEnergy = 0, startImpuls = 0, endImpuls = 0;
	layer.assign(N, Vec3{ 0,0,0 });
	Vec3 a_mean = Vec3{ 0,0,0 };
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
				layer[j] = (v - a * dt + sigma * Randomizer::sampleNormVec3() * sdt);
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
	std::cout << "a mean = " << a_mean.a[0] << "\t" << a_mean.a[1] << "\t" << a_mean.a[2] << endl;
	std::cout << "sigma mean = \n" << sigma_mean.a[0][0] << "\t" << sigma_mean.a[0][1] << "\t" << sigma_mean.a[0][2] << endl <<
		sigma_mean.a[1][0] << "\t" << sigma_mean.a[1][1] << "\t" << sigma_mean.a[1][2] << endl <<
		sigma_mean.a[2][0] << "\t" << sigma_mean.a[2][1] << "\t" << sigma_mean.a[2][2] << endl;
}

Vec3 SemiDiffusionParticleSimulator::demolition(const Vec3& v, double t)
{
	Vec3 integral = Vec3{ 0,0,0 };
	for (int j = 0; j < N; ++j)
	{
		for (int k = 0; k < omegas.size(); ++k)
		{
			Vec3 w = omegas[k];
			Vec3 dv = v - prev_layer[j];
			double prod = qvm::dot(w, dv);
			double lambda = abs(prod) / N / 2 * dt * dw;
			integral += w * qvm::dot(dv, w) * lambda *
				multivariateStandardNormal(prev_layer[j].a[0], prev_layer[j].a[1], prev_layer[j].a[2]);
		}
	}
	return integral * 10;
}

Mat3 SemiDiffusionParticleSimulator::volatility(const Vec3& v, double t)
{
	Mat3 integral = Mat3{ 0,0,0,0,0,0,0,0,0 };
	for (int j = 0; j < N; ++j)
	{
		for (int k = 0; k < omegas.size(); ++k)
		{
			Vec3 w = omegas[k];

			Vec3 dv = v - prev_layer[j];
			double prod = qvm::dot(w, dv);
			double lambda = abs(prod) / N  / 2 * dt * dw;
			Vec3 f = w * qvm::dot(dv, w);
			Mat3 m;
			for (int row = 0; row <= 2; ++row)
				for (int col = 0; col <= 2; ++col)
					m.a[row][col] = f.a[row] * f.a[col];
			integral += m * qvm::mag(dv) * lambda *
				multivariateStandardNormal(prev_layer[j].a[0], prev_layer[j].a[1], prev_layer[j].a[2]);
		}
	}
	integral *= 10;
	arma::mat33 M{integral.a[0][0],integral.a[0][1],integral.a[0][2],integral.a[1][0],integral.a[1][1],integral.a[1][2],integral.a[2][0],integral.a[2][1],integral.a[2][2] };
	arma::mat33  sqrM;
	arma::sqrtmat_sympd(sqrM,M);
	Mat3 res;
	for (int i = 0; i <= 2; ++i)
		for (int j = 0; j <= 2; ++j)
			res.a[i][j] = sqrM.at(i, j);
	return integral;
}