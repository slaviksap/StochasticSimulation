#include <iostream>
#include <fstream>
#include "DiffusionParticleSimulator.h"
#include "SemiDiffusionParticleSimulator.h"
#include "SpasmodicParticleSimulator.h"
#include "OneDimensionalDifferenceProblem.h"
#include <omp.h>

using namespace std;


void SpasmodicSimulation()
{
	int N = 100;
	double t = 1;
	int k = 5000;
	int num_threads = 1;
	int trials_num = 8;
	int trials_for_thread = trials_num / num_threads;
	int sect = 100;
	double min = -5;
	double max = 5;
	vector<double> result(sect,0);
	vector<double> init(sect, 0);
	omp_set_num_threads(num_threads);
#pragma omp parallel
	{
		vector<double> histo(sect,0);
		vector<double> histo_init(sect, 0);
		for (int i = 1; i <= trials_for_thread; ++i)
		{
			SpasmodicParticleSimulator simulator(N, t, k,10, 40,time(0));
			simulator.setInitialDistribution(Distributions::NORM, 0, 1, 1, -5, 5, 100);
			auto start = omp_get_wtime();
			simulator.Calculate();

			cout << "Time: " << omp_get_wtime() - start << endl;

			vector<double>h;
			simulator.make_histogram(sect, min, max, h);
			for (int j = 0; j < sect; ++j)
				histo[j] += h[j];
			vector<double>hi;
			simulator.make_histogram_init(sect, min, max, hi);
			for (int j = 0; j < sect; ++j)
				histo_init[j] += hi[j];
		}
#pragma omp critical
		{
			for (int i = 0; i < sect; ++i)
				result[i] += histo[i];
			for (int i = 0; i < sect; ++i)
				init[i] += histo_init[i];
		}
		if (omp_get_thread_num() == 0)
		{
			for (int i = 0; i < sect; ++i)
				result[i] /= (trials_for_thread * num_threads);
			ofstream file;
			file.open("Tables\\last_spasmodic.txt");
			const double dx = (max - min) / sect;
			for (int i = 0; i < sect; ++i)
			{
				file << min + dx * i << "\t" << result[i] << endl;
			}
			file.close();

			file.open("Tables\\init_spasmodic.txt");
			for (int i = 0; i < sect; ++i)
				init[i] /= (trials_for_thread * num_threads);
			for (int i = 0; i < sect; ++i)
			{
				file << min + dx * i << "\t" << init[i] << endl;
			}
			file.close();
		}
	}
}

void SpasmodicRungeOOC()
{
	int N = 800;
	int k = 100;
	int M = 200;
	double t = 4;
	int gridsize = 1200;
	vector<double> u1;
	{
		SpasmodicParticleSimulator simulator(N, t, k, M, 40, rand());
		simulator.setInitialDistribution(Distributions::NORM, 0, 45, 1, -220, 220, gridsize);
		simulator.Calculate();
		u1 = simulator.result;
	}
	vector<double> u2;
	t /= 2;
	{
		SpasmodicParticleSimulator simulator(N, t, k, M, 40, rand());
		simulator.setInitialDistribution(Distributions::NORM, 0, 45, 1, -220, 220, gridsize);
		simulator.Calculate();
		u2 = simulator.result;
	}
	vector<double> u3;
	t /= 2;
	{
		SpasmodicParticleSimulator simulator(N, t, k, M, 40, rand());
		simulator.setInitialDistribution(Distributions::NORM, 0, 45, 1, -220, 220, gridsize);
		simulator.Calculate();
		u3 = simulator.result;
	}
	vector<double> du1(gridsize);
	vector<double> du2(gridsize);
	for (int i = 0; i < u1.size(); ++i)
	{
		du1[i] = u2[i] - u1[i];
		du2[i] = u3[i] - u2[i];
	}
	double eps1 = dispersion(du1);
	double eps2 = dispersion(du2);
	cout << "Order of approximation in disp norm for N = " << log2(eps1 / eps2) << endl;
	eps1 = L2norm(du1);
	eps2 = L2norm(du2);
	cout << "Order of approximation in L2 norm for N = " << log2(eps1 / eps2) << endl;
}
int main()
{
	srand(time(0));
	Randomizer::init(8, rand(), 100000);
	SpasmodicParticleSimulator simulator (500, 1, 100, 400, 40, rand());
	simulator.setInitialDistribution(Distributions::RECTANGLE, 0, 45, 1, -220, 220, 1200);
	simulator.Calculate();
	simulator.write_result("Tables\\last_spasmodic.txt");
	simulator.write_init("Tables\\init_spasmodic.txt");
	//SpasmodicRungeOOC();
	cout << "Ready!\n";
	cin.get();
	return 0;
}