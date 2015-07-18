// Spiking Neural Net.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Net.h"
#include "TrainData.h"

int _tmain(int argc, _TCHAR* argv[])
{
	TrainData trainData("trainingData.txt");
	vector<unsigned> topology;
	trainData.getTopology(topology);

	Net network(topology);
	vector<double> inputValues, targetValues, resultValues;
	int trainingPass = 0;

	while (!trainData.isEof()) {
		++trainingPass;
		cout << endl << "Pass " << trainingPass;

		// Get new input data and feed it forward:
		if (trainData.getNextInputs(inputValues) != topology[0]) {
			break;
		}

		network.showVectorValues(": Inputs:", inputValues);
		network.feedForward(inputValues);

		// Collect the net's actual output results:
		network.getResults(resultValues);
		network.showVectorValues("Outputs:", resultValues);

		// Train the net what the outputs should have been:
		trainData.getTargetOutputs(targetValues);
		network.showVectorValues("Targets:", targetValues);
		assert(targetValues.size() == topology.back());

		network.backPropagate(targetValues);

		// Report how well the training is working, averaged 
		cout << endl << "The running average error is: " << network.getRecentTrainingError() << endl;

	}
	cout << "Num neurons: " << network.neurons << endl;

	cout << endl << "Finished training the network." << endl;
	cout << "Network accuracy for " << trainingPass - 2 << " training examples is: ~" << 100 - network.getRecentTrainingError() << "%" << endl;


	return 0;
}

