#include "stdafx.h"
#include "Net.h"

double Net::m_recentAverageSmoothingFactor = 100; // The number of training samples to average over.

Net::Net(const vector<unsigned>& topology) 
	: numLayers(topology.size())
{

	// Create layers.
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)	{
		m_layers.push_back(Layer());
		cout << endl << "Made a new layer! The layer will contain " << topology[layerNum] << " neuron(s) + a bias neuron." << endl;

		// Get the amount of outputs that each neuron in this layer has. Output neurons get 0.
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		// Fill the layer with neurons. Using <= to add an extra bias neuron.
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)	{
			// Get the last layer.
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			cout << "Made a new neuron!" << endl;
			neurons++;
		}

		// Force bias node's output value to 1.0;
		m_layers.back().back().setOutputValue(1.0);
	}
}


void Net::feedForward(const vector<double> &inputValues)
{
	assert(inputValues.size() == m_layers[0].size() - 1);

	// Set input values in input neurons (the first layer only).
	for (unsigned i = 0; i < inputValues.size(); ++i)	{
		m_layers[0][i].setOutputValue(inputValues[i]);
	}

	// Feed forward, starting at the first hidden layer.
	for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {

		// Get the previous layer.
		Layer &previousLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
			m_layers[layerNum][n].Neuron::feedForward(previousLayer);
		}
	}
}


void Net::backPropagate(const vector<double>& targetValues)
{
	// Calculate overall (RMS) net error.
	// Get the net's output layers.
	Layer &outputLayer = m_layers.back();
	m_error = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
		double delta = targetValues[n] - outputLayer[n].getOutputValue();
		m_error += delta * delta;
	}
	m_error /= outputLayer.size() - 1; // get average error squared
	m_error = sqrt(m_error); // RMS


	// Get the running average.
	m_recentAverageError =
		(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
		/ (m_recentAverageSmoothingFactor + 1.0);

	// Calculate output layer gradients.
	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
		outputLayer[n].calculateOutputGradient(targetValues[n]);
	}


	// Calculate input layer gradients.
	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];
		
		for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
			hiddenLayer[n].calculateHiddenGradient(nextLayer);
		}
	}

	// Update connection weights for all layers except the input layer and update connection weights.
	// Starts from output layer backwards.
	for (unsigned layerNum{ m_layers.size() - 1 }; layerNum > 0; --layerNum) {
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n) {
			layer[n].updateInputWeights(prevLayer);
		}
	}
}


void Net::getResults(vector<double> &resultValues) const
{
	resultValues.clear();

	for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
		resultValues.push_back(m_layers.back()[n].getOutputValue());
	}
}


void Net::showVectorValues(string label, vector<double>& v)
{
	cout << label << " ";
	for (unsigned i = 0; i < v.size(); ++i) {
		cout << v[i] << " ";
	}
	cout << endl;
}


double Net::getRecentTrainingError()
{
	return m_recentAverageError;
}
