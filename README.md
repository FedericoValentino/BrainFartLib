

# BrainFart
BrainFart is a C++ library for DeepNeuralNetwork development. It's usage is very simple and intuitive.    
To start you will need to create a brain object:


     BrainFart* MyBrain = new BrainFart({2, 3, 2}, 0.05);  

This will create a Neural Network with 2 inputs, 1 hidden layer and 2 outputs and a learining rate of 0.05.  
To feed it data we use the feedForward function:


     MyBrain->feedForward({1, 1});  

This returns a float array with what our network thinks is the answer.  
If we want to train the network we can use the smart and easy backPropagation function:


     MyBrain->backPropagation({actualAnswerVector}, {networksGuessVector});  

The library also offers a train function which takes as input a struct trainingData:

	 struct TrainingData{
		 std::vector<float> Data;
		 std::vector<float> answer;
		 }
	
	MyBrain->train(MyTrainingData);


We can mutate our current Neural Network through the mutate() function and also make two Neural Networks reproduce and make babies through the reproduce function:

    MyBrain->mutate();
    BrainFart* MyBrain2 = new BrainFart({2, 3, 2}, 0.05);
    BrainFart* Brain3 = BrainFart::reproduce(MyBrain, MyBrain2);


The library is still a WIP.