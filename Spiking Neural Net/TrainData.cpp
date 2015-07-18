#include "stdafx.h"
#include "TrainData.h"

TrainData::TrainData(const string inputFile)
{
	m_trainingDataFile.open(inputFile.c_str());
}


void TrainData::getTopology(vector<unsigned> &topology)
{
	string line;
	string label;

	getline(m_trainingDataFile, line);
	stringstream ss(line);
	ss >> label;
	if (this->isEof() || label.compare("topology:") != 0) {
		abort();
	}

	while (!ss.eof()) {
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}

	return;
}

unsigned TrainData::getNextInputs(vector<double> &inputValues)
{
	inputValues.clear();

	string line;
	getline(m_trainingDataFile, line);
	stringstream ss(line);

	string label;
	ss >> label;
	if (label.compare("in:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			inputValues.push_back(oneValue);
		}
	}

	return inputValues.size();
}

unsigned TrainData::getTargetOutputs(vector<double> &targetOutputValues)
{
	targetOutputValues.clear();

	string line;
	getline(m_trainingDataFile, line);
	stringstream ss(line);

	string label;
	ss >> label;
	if (label.compare("out:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			targetOutputValues.push_back(oneValue);
		}
	}

	return targetOutputValues.size();
}
