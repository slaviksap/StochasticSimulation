#pragma once
#include "ParticleSimulator.h"
class SemiDiffusionParticleSimulator : public ParticleSimulator
{
	double E = 0;
public:
	SemiDiffusionParticleSimulator(int N, double t, size_t k, int M) : ParticleSimulator(N, t, k, M) {}
	void Calculate();
	Vec3 demolition(const Vec3& v, double t);
	Mat3 volatility(const Vec3& v, double t);
	void CalculateEnergy();
	double CalculateLayerEnergy();
	double dE();
};

