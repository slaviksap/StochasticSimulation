#include "SemiDiffusionParticleSimulator.h"

using namespace boost::qvm;
const double Pi = 3.1415926535897;
void SemiDiffusionParticleSimulator::Calculate()
{
	CalculateEnergy();
	cout << "E0 = " << E << endl;
	prev_layer = first_layer;
	double t = 0;
	for (int i = 1; i <= k; ++i)
	{
		t += dt;
		layer.assign(N, Vec3{ 0,0,0 });
#pragma omp parallel for
		for (int j = 0; j < N; ++j)
		{
			Vec3 v = prev_layer[j];
			layer[j] = (v - demolition(v, t) * dt + volatility(v, t) * Randomizer::sampleNormVec3() * sdt);
		}
		double de = dE();
		for (int j = 0; j < N; ++j)
		{
			/*double v = mag_sqr(prev_layer[j]);
			if (v + de < 0)
			{
				cout << "Big de, negative square root\n";
				return;
			}*/
			//layer[j] *= sqrt(de);
		}
		if (i != k)
			prev_layer = move(layer);
	}
	cout << "E last = " << CalculateLayerEnergy() << endl;
}

Vec3 SemiDiffusionParticleSimulator::demolition(const Vec3& v, double t)
{
	Vec3 sum = Vec3{ 0,0,0 };
	for (int j = 0; j < N; ++j)
	{
		auto direction = v - prev_layer[j];
		sum += direction * mag(direction);
	}
	return sum / N * (Pi / 2);
}

Mat3 SemiDiffusionParticleSimulator::volatility(const Vec3& v, double t)
{
	Mat3 mat = zero_mat<double, 3>();
	for (int i = 0; i < 3; ++i)
	{
		double sum = 0;
		for (int j = 0; j < N; ++j)
		{
			double len = mag(v - prev_layer[j]);
			sum += pow(len, 3) / 3 + pow(v.a[i] - prev_layer[j].a[i], 2) * len;
		}
		mat.a[i][i] = sqrt(sum / N / (4 * sqrt(Pi)));
	}
	return mat;
}

void SemiDiffusionParticleSimulator::CalculateEnergy()
{
	for (int i = 0; i < N; ++i)
		E += mag_sqr(first_layer[i]);
}
double SemiDiffusionParticleSimulator::CalculateLayerEnergy()
{
	double res = 0;
	for (auto& v : layer)
		res += mag_sqr(v);
	return res;
}

double SemiDiffusionParticleSimulator::dE()
{
	double sum = 0;
	for (int i = 0; i < N; ++i)
	{
		sum += mag_sqr(layer[i]);
	}
	double sum2 = 0;
	for (int i = 0; i < N; ++i)
	{
		sum2 += mag_sqr(prev_layer[i]);
	}
	double de = sum2/sum;
	return de;
}
