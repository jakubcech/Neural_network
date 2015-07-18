#include "stdafx.h"
#include "Neuron.h"

double Neuron::eta = 0.15; // The overall net training rate [0.0, ..., 1.0].
double Neuron::momentumConstant = 0.6; // Momentum

Neuron::Neuron(unsigned numOutputs, unsigned neuronIndex)
{
	for (unsigned c = 0; c < numOutputs; ++c)
	{
		m_outputWeights.push_back(Connection());

		// TODO: Make weights into a class.
		m_outputWeights.back().weight = randomWeight();
	}

	m_neuronIndex = neuronIndex;
}


void Neuron::feedForward(const Layer &previousLayer)
{
	double sum = 0;

	// Sum the previous layer's outputs. Include the bias node from the previous layer.
	for (unsigned n = 0; n < previousLayer.size(); ++n)	{
		sum += previousLayer[n].getOutputValue() * previousLayer[n].m_outputWeights[m_neuronIndex].weight;
	}

	m_outputValue = Neuron::activationFunction(sum);

}


void Neuron::calculateOutputGradient(double targetValue)
{
	double delta = targetValue - m_outputValue;
	m_gradient = delta * Neuron::activationFunctionDerivative(m_outputValue);
}


void Neuron::calculateHiddenGradient(const Layer &nextLayer)
{
	double sumD = sumDerivatives(nextLayer);
	m_gradient = sumD * Neuron::activationFunctionDerivative(m_outputValue);
}


double Neuron::sumDerivatives(const Layer &nextLayer) const
{
	double sum = 0.0;
	for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}

	return sum;
}


void Neuron::updateInputWeights(Layer & previousLayer)
{
	// Update the weights in the Connection container in the neurons of the preceding layer.
	for (unsigned n = 0; n < previousLayer.size(); ++n) {

		Neuron &neuron = previousLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_neuronIndex].deltaWeight;

		double newDeltaWeight =
			eta 
			* neuron.getOutputValue() 
			* m_gradient
			// Add a momentum constant which is equal to a fraction of the previous delta weight. This helps the network to not get stuck in a local minima.
			+ momentumConstant 
			* oldDeltaWeight;

		neuron.m_outputWeights[m_neuronIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_neuronIndex].weight += newDeltaWeight;
	}
}


double Neuron::activationFunction(double sum)
{
	// Tanh derivative approximation. Needs to be rewritten to a proper derivative for real scenarios.
	return tanh(sum);
}


double Neuron::activationFunctionDerivative(double sum)
{
	// Output range of [-1.0, ..., 1.0]
	return 1.0 - sum * sum;
}


void Neuron::setOutputValue(double value)
{
	m_outputValue = value;
}


double Neuron::getOutputValue() const
{
	return m_outputValue;
}