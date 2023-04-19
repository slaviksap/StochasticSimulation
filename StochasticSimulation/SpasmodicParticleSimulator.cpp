#include "SpasmodicParticleSimulator.h"
#include <omp.h>
using namespace boost::qvm;
using namespace std;
void SpasmodicParticleSimulator::Calculate(int trial_num)
{
	vector<Vec3> omegas;
	for (double phi = 0; phi < 2 * _Pi; phi += dPhi)
	{
		for (double thetta = 0; thetta < _Pi; thetta += dThetta)
		{
			omegas.emplace_back(Vec3{ sin(thetta) * cos(phi),sin(thetta) * sin(phi),cos(thetta) });
		}
	}
	dw = 4 * _Pi;
	prev_layer = first_layer;
	layer = first_layer;
	cout << "Start energy = " << Energy(first_layer) << endl;
	cout << "Start impuls = " << Impuls(first_layer) << endl;
	mt19937 gen(start_seed);
	gen.discard(k * N * omp_get_thread_num() + k * N * omp_get_num_threads() * trial_num);
	uniform_real_distribution<>uni(0, 1);
	uniform_int_distribution<>uni_omega(0, omegas.size() - 1);
	auto sampleOmega = [&]() {return omegas[uni_omega(gen)]; };
	auto sampleUni = [&]() {return uni(gen); };
	int count = 0;
	for (int timeStep = 1; timeStep <= k; ++timeStep)
	{
		double lambdamax = 0;
		for (int i = 0; i < N; ++i)
		{
			for (int j = i + 1; j < N; ++j)
			{
				auto w = sampleOmega();
				double prod = boost::qvm::dot(w, prev_layer[i] - prev_layer[j]);
				double lambda = abs(prod) / N / Kn / 2 * dt * dw;
				if (lambda > lambdamax)
					lambdamax = lambda;
				if (sampleUni() < lambda)
				{
					Vec3 dv = w * prod;
					layer[i] -= dv;
					layer[j] += dv;
					++count;
				}
			}
			
		}
		if (timeStep != k)
			prev_layer = layer;
	}
	cout << "End energy = " << Energy(layer) << endl;
	cout << "End impuls = " << Impuls(layer) << endl;
	cout << "Average number of collisions = " << double(count) / k << endl;
}


