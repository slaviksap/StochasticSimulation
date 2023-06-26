#include "SpasmodicParticleSimulator.h"
#include<fstream>
#include<algorithm>
#include <omp.h>
using namespace boost::qvm;
using namespace std;
void SpasmodicParticleSimulator::Calculate()
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
	mt19937 gen(start_seed);
	//gen.discard(k * N * omp_get_thread_num() + k * N * omp_get_num_threads() * trial_num);
	uniform_real_distribution<>uni(0, 1);
	uniform_int_distribution<>uni_omega(0, omegas.size() - 1);
	auto sampleOmega = [&]() {return omegas[uni_omega(gen)]; };
	auto sampleUni = [&]() {return uni(gen); };
	int count = 0;
	ofstream lambda_data;
	lambda_data.open("Tables\\lambda_data.txt");

	//Тестовая плотность
	vector<double> density(N, 0);
	double dRo = 0.1;
	//Статистическая информация
	double startEnergy = 0, endEnergy = 0, startImpuls = 0, endImpuls = 0;
	double maxEchange = 0;
	double avgDV = 0;
	double lambdamax = 0;

	//Основной цикл
	for (int m = 1; m <= M; ++m)
	{
		initialStochasticDistribution();
		prev_layer = first_layer;
		layer = first_layer;
		double t = 0;

		for (int timeStep = 1; timeStep <= k; ++timeStep)
		{
			//Строим гистограмму
			for (int i = 0; i < N; ++i)
			{
				int hcount = 0;
				for (int j = 0; j < N; ++j)
				{
					if (abs(prev_layer[j].a[0] - prev_layer[i].a[0]) < dRo &&
						abs(prev_layer[j].a[1] - prev_layer[i].a[1]) < dRo &&
						abs(prev_layer[j].a[2] - prev_layer[i].a[2]) < dRo)
						++hcount;
				}
				density[i] = double(hcount);
			}
			////////////////
			vector<int> collidingParticles;
			////////////////
			for (int i = 0; i < N; ++i)
			{
				if (find(collidingParticles.begin(), collidingParticles.end(), i) != collidingParticles.end())
					continue;
				for (int j = i + 1; j < N; ++j)
				{
					if (find(collidingParticles.begin(), collidingParticles.end(), j) != collidingParticles.end())
						continue;
					auto w = sampleOmega();
					double prod = boost::qvm::dot(w, prev_layer[i] - prev_layer[j]);
					double lambda = abs(prod) / N / Kn / 2 * dt * dw * density[j];
					//lambda_data << lambda << " ";
					if (lambda > lambdamax)
						lambdamax = lambda;
					if (sampleUni() < lambda)
					{
						Vec3 dv = w * prod;
						avgDV += boost::qvm::mag(dv);
						double E0 = Energy(layer);
						layer[i] -= dv;
						layer[j] += dv;
						double Ediff = Energy(layer) - E0;
						if (Ediff > maxEchange)
							maxEchange = Ediff;
						collidingParticles.push_back(j);
						++count;
						break;
					}
				}

			}
			if (timeStep != k)
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
	std::cout << "Start energy = " << startEnergy << endl << "Start impuls = " << startImpuls << endl;
	std::cout << "End energy = " << endEnergy << endl << "End Impuls = " << endImpuls << endl;
	std::cout << "Energy change = " << startEnergy / endEnergy << endl << "Impuls change = " << startImpuls / endImpuls << endl;
	cout << "Average number of collisions = " << double(count) / k / M << endl;
	cout << "Maximum energy difference = " << maxEchange << endl;
	cout << "Average dv = " << avgDV / count << endl;
}

void SpasmodicParticleSimulator::CalculateRandomNTest(double deviation)
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
	mt19937 gen(start_seed);
	uniform_real_distribution<>uni(0, 1);
	uniform_int_distribution<>uni_omega(0, omegas.size() - 1);
	auto sampleOmega = [&]() {return omegas[uni_omega(gen)]; };
	auto sampleUni = [&]() {return uni(gen); };
	int count = 0;

	//Тестовая плотность
	vector<double> density(N, 0);
	double dRo = 0.1;
	//Статистическая информация
	double startEnergy = 0, endEnergy = 0, startImpuls = 0, endImpuls = 0;
	double maxEchange = 0;
	double avgDV = 0;
	double lambdamax = 0;

	int startN = N;
	//Основной цикл
	for (int m = 1; m <= M; ++m)
	{
		N = startN;
		initialStochasticDistribution();
		prev_layer = first_layer;
		layer = first_layer;
		double t = 0;

		for (int timeStep = 1; timeStep <= k; ++timeStep)
		{
			N = startN / 2 + (Randomizer::sampleUni() * 2 - 1) * (startN / 2) * deviation;
			if (N < 0)
				N = 0;
			int seed = rand();
			mt19937 tempGen(seed);
			shuffle(prev_layer.begin(), prev_layer.end(), tempGen);
			tempGen.seed(seed);
			shuffle(layer.begin(), layer.end(), tempGen);
			//Строим гистограмму
			for (int i = 0; i < N; ++i)
			{
				int hcount = 0;
				for (int j = 0; j < N; ++j)
				{
					if (abs(prev_layer[j].a[0] - prev_layer[i].a[0]) < dRo &&
						abs(prev_layer[j].a[1] - prev_layer[i].a[1]) < dRo &&
						abs(prev_layer[j].a[2] - prev_layer[i].a[2]) < dRo)
						++hcount;
				}
				density[i] = double(hcount);
			}
			////////////////
			vector<int> collidingParticles;
			////////////////
			for (int i = 0; i < N; ++i)
			{
				if (find(collidingParticles.begin(), collidingParticles.end(), i) != collidingParticles.end())
					continue;
				for (int j = i + 1; j < N; ++j)
				{
					if (find(collidingParticles.begin(), collidingParticles.end(), j) != collidingParticles.end())
						continue;
					auto w = sampleOmega();
					double prod = boost::qvm::dot(w, prev_layer[i] - prev_layer[j]);
					double lambda = abs(prod) / N / Kn / 2 * dt * dw * density[j];
					if (lambda > lambdamax)
						lambdamax = lambda;
					if (sampleUni() < lambda)
					{
						Vec3 dv = w * prod;
						avgDV += boost::qvm::mag(dv);
						double E0 = Energy(layer);
						layer[i] -= dv;
						layer[j] += dv;
						double Ediff = Energy(layer) - E0;
						if (Ediff > maxEchange)
							maxEchange = Ediff;
						collidingParticles.push_back(j);
						++count;
						break;
					}
				}

			}
			if (timeStep != k)
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
	std::cout << "Start energy = " << startEnergy << endl << "Start impuls = " << startImpuls << endl;
	std::cout << "End energy = " << endEnergy << endl << "End Impuls = " << endImpuls << endl;
	std::cout << "Energy change = " << startEnergy / endEnergy << endl << "Impuls change = " << startImpuls / endImpuls << endl;
	cout << "Average number of collisions = " << double(count) / k / M << endl;
	cout << "Maximum energy difference = " << maxEchange << endl;
	cout << "Average dv = " << avgDV / count << endl;
}

void SpasmodicParticleSimulator::CalculateDiffLambdaTest()
{
	vector<Vec3> omegas;
	vector<double> lambdaMulty = { 0.001,0.1,0.1,0.1,1,1,1,1,10,10,10,100,100,1000,10000 };
	for (double phi = 0; phi < 2 * _Pi; phi += dPhi)
	{
		for (double thetta = 0; thetta < _Pi; thetta += dThetta)
		{
			omegas.emplace_back(Vec3{ sin(thetta) * cos(phi),sin(thetta) * sin(phi),cos(thetta) });
		}
	}
	dw = 4 * _Pi;
	mt19937 gen(start_seed);
	//gen.discard(k * N * omp_get_thread_num() + k * N * omp_get_num_threads() * trial_num);
	uniform_real_distribution<>uni(0, 1);
	uniform_int_distribution<>uni_omega(0, omegas.size() - 1);
	uniform_int_distribution<>uni_multy(0, lambdaMulty.size() - 1);
	auto sampleOmega = [&]() {return omegas[uni_omega(gen)]; };
	auto sampleUni = [&]() {return uni(gen); };
	auto sampleLambdaMulty = [&]() {return lambdaMulty[uni_multy(gen)]; };
	int count = 0;
	ofstream lambda_data;
	lambda_data.open("Tables\\lambda_data.txt");

	//Тестовая плотность
	vector<double> density(N, 0);
	double dRo = 0.1;
	//Статистическая информация
	double startEnergy = 0, endEnergy = 0, startImpuls = 0, endImpuls = 0;
	double maxEchange = 0;
	double avgDV = 0;
	double lambdamax = 0;

	//Основной цикл
	for (int m = 1; m <= M; ++m)
	{
		initialStochasticDistribution();
		prev_layer = first_layer;
		layer = first_layer;
		double t = 0;

		for (int timeStep = 1; timeStep <= k; ++timeStep)
		{
			//Строим гистограмму
			for (int i = 0; i < N; ++i)
			{
				int hcount = 0;
				for (int j = 0; j < N; ++j)
				{
					if (abs(prev_layer[j].a[0] - prev_layer[i].a[0]) < dRo &&
						abs(prev_layer[j].a[1] - prev_layer[i].a[1]) < dRo &&
						abs(prev_layer[j].a[2] - prev_layer[i].a[2]) < dRo)
						++hcount;
				}
				density[i] = double(hcount);
			}
			////////////////
			vector<int> collidingParticles;
			////////////////
			for (int i = 0; i < N; ++i)
			{
				if (find(collidingParticles.begin(), collidingParticles.end(), i) != collidingParticles.end())
					continue;
				for (int j = i + 1; j < N; ++j)
				{
					if (find(collidingParticles.begin(), collidingParticles.end(), j) != collidingParticles.end())
						continue;
					auto w = sampleOmega();
					double prod = boost::qvm::dot(w, prev_layer[i] - prev_layer[j]);
					double lambda = abs(prod) / N / Kn / 2 * dt * dw * density[j] * sampleLambdaMulty();
					//lambda_data << lambda << " ";
					if (lambda > lambdamax)
						lambdamax = lambda;
					if (sampleUni() < lambda)
					{
						Vec3 dv = w * prod;
						avgDV += boost::qvm::mag(dv);
						double E0 = Energy(layer);
						layer[i] -= dv;
						layer[j] += dv;
						double Ediff = Energy(layer) - E0;
						if (Ediff > maxEchange)
							maxEchange = Ediff;
						collidingParticles.push_back(j);
						++count;
						break;
					}
				}

			}
			if (timeStep != k)
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
	std::cout << "Start energy = " << startEnergy << endl << "Start impuls = " << startImpuls << endl;
	std::cout << "End energy = " << endEnergy << endl << "End Impuls = " << endImpuls << endl;
	std::cout << "Energy change = " << startEnergy / endEnergy << endl << "Impuls change = " << startImpuls / endImpuls << endl;
	cout << "Average number of collisions = " << double(count) / k / M << endl;
	cout << "Maximum energy difference = " << maxEchange << endl;
	cout << "Average dv = " << avgDV / count << endl;
}
