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

float BrainFart::sig(float x) {
    return 1/(1 + std::exp(-x));
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
        layers[i] = MatrixMath::multiply(1, dimensions[i-1], dimensions[i-1], dimensions[i], layers[i-1], weights[i-1]);

        /*
        float sum = 0;
        for(int j = 0; j < dimensions[i]; j++)
        {
            sum += layers[i][0][j];
        }
        */

        if(i != dimensions.size()-1)
        {
            for(int j = 0; j < dimensions[i]; j++)
            {
                layers[i][0][j] = reLU(layers[i][0][j]);
            }
        }
        else
        {
            for(int j = 0; j < dimensions[i]; j++)
            {
                layers[i][0][j] = sig(layers[i][0][j]);
            }
        }
    }

    /*
    printf("Network Output is: \n");
    print(layers[dimensions.size()-1], 1, dimensions[dimensions.size()-1]);
     */

    float* returnValue = new float[dimensions[dimensions.size()-1]];

    for(int i = 0; i < dimensions[dimensions.size()-1]; i++)
    {
        returnValue[i] = layers[dimensions.size()-1][0][i];
    }

    return returnValue;
}

void BrainFart::freeLayers()
{
    for(int i = 0; i < dimensions.size(); i++)
    {
        delete layers[i][0];
        delete layers[i];
    }
}

void BrainFart::backwardPropagation(std::vector<float> actual, std::vector<float> guess)
{
    /*
     * Errors[0] are the errors on the output side, errors[dimensions.size()-2] are the errors on the first hidden layer
     * deltaWeights[0] are the change in weights on the last set of weights, deltaWeights[dimensions.size()-1] are the change in weights on the first set
     */

    float*** errors = new float**[dimensions.size() - 1];
    float*** deltaWeights = new float**[dimensions.size() - 1];

    float** actualMatrix = MatrixMath::toMatrix(1, actual.size(), actual);
    float** guessMatrix = MatrixMath::toMatrix(1, guess.size(), guess);

    errors[0] = MatrixMath::subtract(1, guess.size(), actualMatrix, guessMatrix);

    for(int i = 1; i < dimensions.size() - 1; i++)
    {
        float** weightT = MatrixMath::transpose(weights[dimensions.size()-1-i], dimensions[dimensions.size()-1-i], dimensions[dimensions.size()-i]);
        errors[i] = MatrixMath::multiply(1, dimensions[dimensions.size()-i], dimensions[dimensions.size()-i], dimensions[dimensions.size()-1-i], errors[i-1], weightT);
    }

    for(int i = 0; i < dimensions.size() - 1; i++)
    {
        MatrixMath::Hadamard(1, dimensions[dimensions.size()-1-i], errors[i], layers[dimensions.size()-1-i]);
        float** errorT = MatrixMath::transpose(errors[i], 1, dimensions[dimensions.size()-1-i]);
        deltaWeights[i] = MatrixMath::multiply( dimensions[dimensions.size()-1-i], 1, 1, dimensions[dimensions.size()-2-i], errorT, layers[dimensions.size()-2-i]);
    }

    for(int i = 0; i < dimensions.size()-2; i++)
    {
        MatrixMath::sum(dimensions[i], dimensions[i+1], weights[i], deltaWeights[dimensions.size()-2-i]);
    }
}

void BrainFart::mutate()
{
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_real_distribution<> u{0.0, 1.0};

    for(int i = 0; i < dimensions.size()-1; i++)
    {
        float** matrix = weights[i];
        for(int j = 0; j < dimensions[i]; j++)
        {
            for(int k = 0; k < dimensions[i+1]; k++)
            {
                if(u(gen) > 0.5)
                {
                    std::normal_distribution<> d{matrix[j][k],2};
                    matrix[j][k] = d(gen);
                }
            }
        }
    }
}

BrainFart *BrainFart::reproduce(BrainFart *father, BrainFart *mother) {
    BrainFart* son = new BrainFart(father->dimensions);

    std::random_device rd{};
    std::mt19937 gen{rd()};

    std::uniform_real_distribution<> d{0.0, 1.0};

    for(int i = 0; i < father->dimensions.size()-1; i++)
    {
        float** fatherMatrix = father->weights[i];
        float** motherMatrix = mother->weights[i];
        float** sonMatrix = son->weights[i];

        for(int j = 0; j < father->dimensions[i]; j++)
        {
            for(int k = 0; k < father->dimensions[i+1]; k++)
            {
                if(d(gen) > 0.5)
                {
                    sonMatrix[j][k] = fatherMatrix[j][k];
                }
                else
                {
                    sonMatrix[j][k] = motherMatrix[j][k];
                }
            }
        }
    }
    return son;
}

void BrainFart::freeBrain()
{
    delete layers;

    for(int i = 0; i < dimensions.size()-1; i++)
    {
        for(int j = 0; j < dimensions[i]; j++)
        {
            delete weights[i][j];
        }
        delete weights[i];
    }
    delete weights;
}

BrainFart *BrainFart::cloneBrain(BrainFart *copy) {
    BrainFart* son = new BrainFart(copy->dimensions);


    for(int i = 0; i < copy->dimensions.size()-1; i++)
    {
        float** fatherMatrix = copy->weights[i];
        float** sonMatrix = son->weights[i];

        for(int j = 0; j < copy->dimensions[i]; j++)
        {
            for(int k = 0; k < copy->dimensions[i+1]; k++)
            {
                sonMatrix[j][k] = fatherMatrix[j][k];
            }
        }
    }
    return son;
}




