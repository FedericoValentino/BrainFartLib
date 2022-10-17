#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <random>
#include "BrainFart.h"
#include "../MatrixLib/MatrixLib.h"

float BrainFart::reLU(float x) {
    if(x > 0)
    {
        return x;
    }
    return 0;
}

BrainFart::BrainFart(std::vector<int> layerSizes)
{
    dimensions = layerSizes;

    layers = new float**[layerSizes.size()];

    weights = new float ** [layerSizes.size()-1];

    for(int i = 0; i < layerSizes.size()-1; i++)
    {
        weights[i] = new float * [layerSizes[i]];
        for(int j = 0; j < layerSizes[i]; j++)
        {
            weights[i][j] = new float[layerSizes[i+1]];
        }
    }

    initializeWeights();
}

void BrainFart::initializeWeights()
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    for(int i = 0; i < dimensions.size() - 1; i++)
    {
        int r = dimensions[i];
        int c = dimensions[i+1];
        for(int j = 0; j < r; j++)
        {
            for(int k = 0; k < c; k++)
            {
                weights[i][j][k] = dist(mt);

            }
        }
        //print(weights[i], r, c);
    }
}

float *BrainFart::feedForward(std::vector<float> input)
{
    if(input.size() != dimensions[0])
    {
        printf("Input is not correct size!\n");
        return nullptr;
    }

    layers[0] = new float*[1];
    layers[0][0] = new float[input.size()];

    for(int i = 0; i < input.size(); i++)
    {
        layers[0][0][i] = input[i];
    }

    /*
    printf("You are feeding the network: \n");
    print(layers[0], 1, dimensions[0]);
     */

    for(int i = 1; i < dimensions.size(); i++)
    {
        layers[i] = multiply(1, dimensions[i-1], dimensions[i-1], dimensions[i], layers[i-1], weights[i-1]);

        for(int j = 0; j < dimensions[i]; j++)
        {
            layers[i][0][j] = reLU(layers[i][0][j]);
        }
    }

    /*
    printf("Network Output is: \n");
    print(layers[dimensions.size()-1], 1, dimensions[dimensions.size()-1]);
     */

    return layers[dimensions.size()-1][0];
}



void BrainFart::backwardPropagation(std::vector<float> input) {

}


