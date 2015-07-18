#pragma once
#include "stdafx.h"

#include <assert.h>
#include <vector>
#include <iostream>
#include <string>

#include "Neuron.h"

using namespace std;

class Net
{
public:
	int neurons = 0;


	Net() = delete;
	Net(const vector <unsigned> &topology);
			
	void feedForward(const vector<double> &inputValues);
	void backPropagate(const vector<double> &targetValues);
	void getResults(vector<double> &resultValues) const;
	void showVectorValues(string label, vector<double> &v);
	double getRecentTrainingError();

private:
	unsigned numLayers;
	vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
	double m_error;
	double m_recentAverageError;
	static double m_recentAverageSmoothingFactor;
};