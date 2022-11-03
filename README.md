
# BrainFart
BrainFart is a C++ library for DeepNeuralNetwork development. It's usage is very simple and intuitive.  
To start you will need to create a brain object:

    BrainFart* MyBrain = new BrainFart({2, 3, 2});

This will create a Neural Network with 2 inputs, 1 hidden layer and 2 outputs.
To feed it data we use the feedForward function:

    MyBrain->feedForward({1, 1});

This returns a float array with what our network thinks is the answer.
If we want to train the network we can use the smart and easy backPropagation function:

    MyBrain->backPropagation({actualAnswerVector}, {networksGuessVector});

We can mutate our current NeuralNetwork through the mutate() function and also make two Neural Networks reproduce and make babies!

The library is still a WIP.