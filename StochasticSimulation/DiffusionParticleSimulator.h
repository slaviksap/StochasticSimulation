#pragma once
#include "ParticleSimulator.h"
class DiffusionParticleSimulator : public ParticleSimulator
{
public:
	DiffusionParticleSimulator(int N, double t, size_t k, int M) : ParticleSimulator(N, t, k, M) {}
	void Calculate();
	double demolition(const Vec3& v, double t);
	Mat3 volatility(const Vec3& v, double t);
};

