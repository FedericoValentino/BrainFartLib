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

    layerNumber = dimensions.size();

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
    for(int i = 0; i < layerNumber - 1; i++)
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

    for(int i = 1; i < layerNumber; i++)
    {
        layers[i] = MatrixMath::multiply(1, dimensions[i-1], dimensions[i-1], dimensions[i], layers[i-1], weights[i-1]);


        if(i != layerNumber-1)
        {
            for(int j = 0; j < dimensions[i]; j++)
            {
                layers[i][0][j] = sig(layers[i][0][j]);
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

    float* returnValue = new float[dimensions[layerNumber-1]];

    for(int i = 0; i < dimensions[layerNumber-1]; i++)
    {
        returnValue[i] = layers[layerNumber-1][0][i];
    }

    return returnValue;
}

void BrainFart::freeLayers()
{
    for(int i = 0; i < layerNumber; i++)
    {
        delete layers[i][0];
        delete layers[i];
    }
}

void BrainFart::backwardPropagation(std::vector<float> actual, std::vector<float> guess)
{

    int dimension = layerNumber - 1;

    float*** errors = new float**[dimension];
    float*** deltaWeights = new float**[dimension];

    float** actualMatrix = MatrixMath::toMatrix(1, actual.size(), actual);
    float** guessMatrix = MatrixMath::toMatrix(1, guess.size(), guess);

    //Errors per layer
    for(int i = dimension-1; i >= 0; i--)
    {
        if(i == dimension - 1)
        {
            errors[i] = MatrixMath::subtract(1, guess.size(), actualMatrix, guessMatrix);
        }
        else
        {
            float** weightT = MatrixMath::transpose(weights[i+1], dimensions[i+1], dimensions[i+2]);
            errors[i] = MatrixMath::multiply(1, dimensions[i+2], dimensions[i+2], dimensions[i+1], errors[i+1], weightT);
        }
    }

    //Gradient and Weight
    for(int i = 0; i < dimension; i++)
    {
        float** gradient = MatrixMath::dsigmoid(1, dimensions[i+1], layers[i+1]);
        MatrixMath::Hadamard(1, dimensions[i+1], gradient, errors[i]);
        float** gradientT = MatrixMath::transpose(gradient, 1, dimensions[i+1]);
        deltaWeights[i] = MatrixMath::multiply(dimensions[i+1], 1, 1, dimensions[i], gradientT, layers[i]);
        deltaWeights[i] = MatrixMath::transpose(deltaWeights[i], dimensions[i+1], dimensions[i]);
        //MatrixMath::print(deltaWeights[i], dimensions[i], dimensions[i+1]);
    }


    for(int i = 0; i < dimension; i++)
    {
        MatrixMath::sum(dimensions[i], dimensions[i+1], weights[i], deltaWeights[i]);

        MatrixMath::print(deltaWeights[i], dimensions[i], dimensions[i+1]);
        MatrixMath::print(weights[i], dimensions[i], dimensions[i+1]);
    }
}

void BrainFart::mutate()
{
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_real_distribution<> u{0.0, 1.0};

    for(int i = 0; i < layerNumber-1; i++)
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

    for(int i = 0; i < father->layerNumber-1; i++)
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

    for(int i = 0; i < layerNumber-1; i++)
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


    for(int i = 0; i < copy->layerNumber-1; i++)
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




