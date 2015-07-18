#pragma once
#include "stdafx.h"

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace std;

class Neuron;

using Layer = vector<Neuron>;

struct Connection
{
	double weight;
	double deltaWeight;
};


class Neuron
{
public:
	Neuron() = delete;
	Neuron(unsigned numOutputs, unsigned neuronIndex);
	void setOutputValue(double value);
	double getOutputValue() const;
	void feedForward(const Layer &previousLayer);

	void calculateOutputGradient(double targetValue);
	void calculateHiddenGradient(const Layer &nextLayer);
	double sumDerivatives(const Layer &nextLayer) const;
	void updateInputWeights(Layer &prevLayer);


	// The membrane potential of the neuron. The neuron only fires (propagates to other neurons once it reaches a certain potential.
	double neuronPotential;

private:
	double m_outputValue;
	vector<Connection> m_outputWeights;
	unsigned m_neuronIndex;
	double m_gradient;
	
	// Neuron activation functions.
	static double activationFunction(double x);
	static double activationFunctionDerivative(double x);

	static double eta; // The overall net training rate [0.0, ..., 1.0].
	static double momentumConstant; // Multiplies the last weight change by a [0.0, ..., 1] value (alpha).

	static double randomWeight(void) 
	{
		return rand() / double(RAND_MAX);
	}
};
