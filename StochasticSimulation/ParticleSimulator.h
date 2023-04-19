#pragma once
#include <iostream>
#include <vector>
#include <math.h>
#include <random>
#include <time.h>
#include "Distributions.h"
#include <boost/qvm.hpp>
#include <omp.h>

using namespace std;
using Vec3 = boost::qvm::vec<double, 3>;
using Mat3 = boost::qvm::mat<double, 3, 3>;

namespace Randomizer
{
	static vector<mt19937> gens(8,mt19937(23232));
	static normal_distribution<> distr(0, 1);
	static uniform_real_distribution<> uni(0, 1);

	static void init(int threads_num, int seed, int jump)
	{
		//gens.assign(threads_num, mt19937(seed));
		for (int i = 0; i < threads_num; ++i)
			gens[i].seed(seed);
		for (int i = 0; i < threads_num; ++i)
			gens[i].discard(jump * i);
	}
	static double sampleNorm()
	{
		return distr(gens[omp_get_thread_num()]);
	}
	static double sampleUni()
	{
		return uni(gens[omp_get_thread_num()]);
	}
	static Vec3 sampleNormVec3()
	{
		Vec3 v;
		int threadnum = omp_get_thread_num();
		v.a[0] = distr(gens[threadnum]);
		v.a[1] = distr(gens[threadnum]);
		v.a[2] = distr(gens[threadnum]);
		return v;
	}
};
class ParticleSimulator
{
public:
	int N;						//число частиц
	int k;						//число шагов по времени
	int M;						//число испытаний
	double tmax;				//интервал по времени
	double dt;					//шаг по времени
	double sdt;					//корень из шага по времени
	double mass;				//интеграл функции
	vector<Vec3> first_layer;	//начальный слой
	vector<Vec3> prev_layer;	//предыдущий временный слой
	vector<Vec3> layer;			//текущий слой по времени
	vector<double> init_result;	//осредненный по всем испытаниям начальные условия
	vector<double> result;		//осредненный по всем испытаниям результат
	ParticleSimulator(int N, double t, size_t k, int M) :N(N), tmax(t),
		k(k),M(M), dt(tmax / k), init(), mass(), prev_layer(), layer()
	{
		sdt = sqrt(dt);
	}
	struct InitDistrib
	{
		Distributions type;
		double a;
		double b;
		double c;
		double min;
		double max;
		int sect;
	};
	InitDistrib init;
	//начальное распределение
	void setInitialDistribution(Distributions type, double a, double b, double c, double min, double max, int sect);
	void initialStochasticDistribution();
	void initialDiscreteDistribution();

	void write_init(const string& fileName);
	void write_result(const string& fileName);
	void write_full_last(const string& fileName);
	void write_full_first(const string& fileName);
	double Energy(vector<Vec3>& layer);
	double Impuls(vector<Vec3>& layer);
	void make_histogram(int sect, double min, double max, vector<double>& hist);
	void make_histogram_init(int sect, double min, double max, vector<double>& hist);
	void replenish_histogram(vector<double>& result, vector<Vec3>& layer);
};

