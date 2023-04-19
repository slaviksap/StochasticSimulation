#pragma once
#include "ParticleSimulator.h"
class SpasmodicParticleSimulator : public ParticleSimulator
{
public:
    double Kn = 1;
    double dw;
    double dPhi;
    double dThetta;
    int angleSteps;
    int start_seed;
    SpasmodicParticleSimulator(int N, double t, size_t k, int M, int angleSteps, int start_seed) : ParticleSimulator(N, t, k, M),
        angleSteps(angleSteps),start_seed(start_seed)
    {
        dPhi = 2 * _Pi / angleSteps;
        dThetta = dPhi;
        dw = 4 * _Pi / (angleSteps * angleSteps / 2);
    }
    void Calculate(int trial_num);
    Vec3 demolition(const Vec3& v, double t){ return Vec3{ 0,0,0, }; }
    Mat3 volatility(const Vec3& v, double t) { return boost::qvm::zero_mat<double, 3>(); }

};

