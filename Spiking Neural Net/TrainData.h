#pragma once


#include "stdafx.h"

#include <assert.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cmath>
#include <string>



using namespace std;

class TrainData {
public:
	TrainData(const string filename);
	bool isEof(void) { return m_trainingDataFile.eof(); }
	void getTopology(vector<unsigned> &topology);

	unsigned getNextInputs(vector<double> &inputVals);
	unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
	ifstream m_trainingDataFile;
};
