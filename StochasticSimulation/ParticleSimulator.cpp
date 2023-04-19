#include "ParticleSimulator.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <memory>
#include <functional>
#include <fstream>

void ParticleSimulator::setInitialDistribution(Distributions type, double a, double b, double c, double min, double max, int sect)
{
	init.type = type;
	init.a = a;
	init.b = b;
	init.c = c;
	init.min = min;
	init.max = max;
	init.sect = sect;
	init_result.assign(sect, 0);
	result.assign(sect, 0);
	//initialStochasticDistribution();
}
void ParticleSimulator::initialDiscreteDistribution()
{
	first_layer.clear();
	if (init.type == Distributions::RECTANGLE)
	{
		double integral = pow((init.b - init.a), 3);
		double mass = integral / N;
		double dx, dy, dz;
		dx = dy = dz = 0.01;
		double dI = 0;
		for (double z = init.a + 0.001; z <= init.b; z += dz)
		{
			for (double y = init.a + 0.001; y <= init.b; y += dy)
			{
				for (double x = init.a + 0.001; x <= init.b; x += dx)
				{
					dI += dx * dy * dz;
					if (dI >= mass)
					{
						first_layer.push_back(Vec3{ x,y,z });
						dI -= mass;
					}
				}
			}
		}
		N = first_layer.size();
		cout << "Real particles number: " << first_layer.size() << endl;
	}
}

void ParticleSimulator::initialStochasticDistribution()
{
	first_layer.clear();
	if (init.type == Distributions::NORM)
	{
		normal_distribution<> norm(init.a, init.b);
		for (int i = 0; i < N; ++i)
		{
			//first_layer.emplace_back(Vec3{ Randomizer::sampleNorm(),0 ,0});
			first_layer.emplace_back(Randomizer::sampleNormVec3());
		}
	}
	if (init.type == Distributions::RECTANGLE)
	{
		double range = init.b - init.a;
		for (int i = 0; i < N; ++i)
		{
			Vec3 v;
			v.a[0] = init.a + Randomizer::sampleUni() * range;
			v.a[1] = init.a + Randomizer::sampleUni() * range;
			v.a[2] = init.a + Randomizer::sampleUni() * range;
			first_layer.push_back(v);
		}
	}
}

void ParticleSimulator::write_full_last(const string& fileName)
{
	ofstream file;
	file.open(fileName);
	for (auto& v : layer)
	{
		file << v.a[0] << "\t" << v.a[1] << "\t" << v.a[2] << endl;
	}
	file.close();
}
void ParticleSimulator::write_full_first(const string& fileName)
{
	ofstream file;
	file.open(fileName);
	for (auto& v : first_layer)
	{
		file << v.a[0] << "\t" << v.a[1] << "\t" << v.a[2] << endl;
	}
	file.close();
}

void ParticleSimulator::make_histogram(int sect, double min, double max, vector<double>& hist)
{
	hist.assign(sect, 0);
	const double dx = (max - min) / sect;
	for (const auto& v : layer)
	{
		double x = v.a[0];
		if (min <= x && x <= max)
			hist[size_t((x - min) / dx)] += 1;
	}
	for (double& x : hist)
		x = x / N / dx;
}
void ParticleSimulator::make_histogram_init(int sect, double min, double max, vector<double>& hist)
{
	hist.assign(sect, 0);
	const double dx = (max - min) / sect;
	for (const auto& v : first_layer)
	{
		double x = v.a[0];
		if (min <= x && x <= max)
			hist[size_t((x - min) / dx)] += 1;
	}
	for (double& x : hist)
		x = x / N / dx;
}

void ParticleSimulator::replenish_histogram(vector<double>& result, vector<Vec3>& layer)
{
	vector<double> hist(init.sect,0);
	const double dx = (init.max - init.min) / init.sect;
	for (const auto& v : layer)
	{
		double x = v.a[0];
		if (init.min <= x && x <= init.max)
			hist[size_t((x - init.min) / dx)] += 1;
	}
	for (double& x : hist)
		x = x / N / dx;
	for (int i = 0; i < result.size(); ++i)
		result[i] += hist[i];
}

void ParticleSimulator::write_init(const string& fileName)
{
	const double dx = (init.max - init.min) / init.sect;
	ofstream file;
	file.open(fileName);
	for (int i = 0; i < init.sect; ++i)
	{
		file << init.min + dx * i << "\t" << init_result[i] << endl;
	}
}
void ParticleSimulator::write_result(const string& fileName)
{
	const double dx = (init.max - init.min) / init.sect;
	ofstream file;
	file.open(fileName);
	for (int i = 0; i < init.sect; ++i)
	{
		file << init.min + dx * i << "\t" << result[i] << endl;
	}
}
double ParticleSimulator::Energy(vector<Vec3>& layer)
{
	double E = 0;
	for (int i = 0; i < layer.size(); ++i)
	{
		E += boost::qvm::mag_sqr(layer[i]) / 2;
	}
	return E;
}

double ParticleSimulator::Impuls(vector<Vec3>& layer)
{
	Vec3 p = { 0,0,0 };
	for (int i = 0; i < layer.size(); ++i)
	{
		p += layer[i];
	}
	return boost::qvm::mag(p);
}