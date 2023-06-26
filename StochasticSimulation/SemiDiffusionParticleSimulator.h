#pragma once
#include "ParticleSimulator.h"
class SemiDiffusionParticleSimulator : public ParticleSimulator
{
	double E = 0;
	vector<Vec3> omegas;
	vector<double> thettas;
	int angleSteps;
	double dPhi;
	double dThetta;
	double dw;
public:
	SemiDiffusionParticleSimulator(int N, double t, size_t k, int M) : ParticleSimulator(N, t, k, M) 
	{
		angleSteps = 10;
		dPhi = 2 * _Pi / angleSteps;
		dThetta = dPhi;
		dw = 4 * _Pi / (angleSteps * angleSteps / 2);
		for (double phi = 0; phi < 2 * _Pi; phi += dPhi)
		{
			for (double thetta = 0; thetta < _Pi; thetta += dThetta)
			{
				omegas.emplace_back(Vec3{ sin(thetta) * cos(phi),sin(thetta) * sin(phi),cos(thetta) });
				thettas.emplace_back(thetta);
			}
		}
	}
	void Calculate();
	Vec3 demolition(const Vec3& v, double t);
	Mat3 volatility(const Vec3& v, double t);
};

