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

BrainFart::BrainFart(std::vector<int> layerSizes, float LR)
{
    learningRate = LR;

    dimensions = layerSizes;

    layerNumber = dimensions.size();

    layers = new float**[layerSizes.size()];

    biases = new float**[layerSizes.size()];

    weights = new float ** [layerSizes.size()-1];

    for(int i = 0; i < layerSizes.size()-1; i++)
    {
        weights[i] = new float * [layerSizes[i+1]];
        for(int j = 0; j < layerSizes[i+1]; j++)
        {
            weights[i][j] = new float[layerSizes[i]];
        }
    }

    initializeWeightsAndBiases();
}

void BrainFart::initializeWeightsAndBiases()
{
    std::random_device rd;
    std::mt19937 mt(rd());
    for(int i = 0; i < layerNumber - 1; i++)
    {
        int r = dimensions[i+1];
        int c = dimensions[i];
        for(int j = 0; j < r; j++)
        {
            for(int k = 0; k < c; k++)
            {
                std::normal_distribution<double> distribution{0,static_cast<float>((2/((dimensions.at(i)+dimensions.at(i+1))))^2)};
                weights[i][j][k] = distribution(mt);
            }
        }
    }

    for(int i = 0; i < layerNumber; i++)
    {
        biases[i] = new float*[dimensions[i]];
        for(int j = 0; j < dimensions[i]; j++)
        {
            std::normal_distribution<double> distribution{0,static_cast<float>((2/((dimensions.at(i)+dimensions.at(i+1))))^2)};
            biases[i][j] = new float[1];
            biases[i][j][0] = distribution(mt);
        }
    }

}

std::vector<float> BrainFart::feedForward(std::vector<float> input)
{
    if(input.size() != dimensions[0])
    {
        printf("Input is not correct size!\n");
        std::vector<float> NULLANSWER;
        return NULLANSWER;
    }

    layers[0] = new float*[dimensions[0]];

    for(int i = 0; i < input.size(); i++)
    {
        layers[0][i] = new float[1];
        layers[0][i][0] = input[i];
    }

    for(int i = 1; i < layerNumber; i++)
    {
        layers[i] = MatrixMath::multiply(dimensions[i], dimensions[i-1], dimensions[i-1], 1, weights[i-1], layers[i-1]);

        MatrixMath::sum(dimensions[i], 1, layers[i], biases[i]);


        //Hidden Layers
        if(i != layerNumber-1)
        {
            for(int j = 0; j < dimensions[i]; j++)
            {
                layers[i][j][0] = sig(layers[i][j][0]);
            }
        }
        //Output Layer
        else
        {
            for(int j = 0; j < dimensions[i]; j++)
            {
                layers[i][j][0] = sig(layers[i][j][0]);
            }
        }
    }

    std::vector<float> returnValue;

    for(int i = 0; i < dimensions[layerNumber-1]; i++)
    {
        returnValue.push_back(layers[layerNumber-1][i][0]);
    }

    return returnValue;
}

void BrainFart::freeLayers()
{
    for(int i = 0; i < layerNumber; i++)
    {
        for(int j = 0; j < dimensions[i]; j++)
        {
            delete layers[i][j];
        }
        delete layers[i];
    }
}

void BrainFart::backwardPropagation(const std::vector<float>& actual, const std::vector<float>& guess)
{

    int dimension = layerNumber - 1;

    auto*** errors = new float**[dimension];
    auto*** deltaWeights = new float**[dimension];

    float** actualMatrix = MatrixMath::toMatrix(actual.size(), 1, actual);
    float** guessMatrix = MatrixMath::toMatrix(guess.size(), 1, guess);

    //Errors per layer
    for(int i = dimension-1; i >= 0; i--)
    {
        if(i == dimension - 1)
        {
            errors[i] = MatrixMath::subtract(guess.size(), 1, actualMatrix, guessMatrix);
            MatrixMath::freeMatrix(actual.size(), 1, actualMatrix);
            MatrixMath::freeMatrix(guess.size(), 1, guessMatrix);
        }
        else
        {
            float** weightT = MatrixMath::transpose(weights[i+1], dimensions[i+2], dimensions[i+1]);
            errors[i] = MatrixMath::multiply(dimensions[i+1], dimensions[i+2], dimensions[i+2], 1, weightT, errors[i+1]);

            MatrixMath::freeMatrix(dimensions[i+1], dimensions[i+2], weightT);
        }
    }

    //Gradient and Weight
    for(int i = 0; i < dimension; i++)
    {
        float** gradient = MatrixMath::dsigmoid(dimensions[i+1], 1, layers[i+1]);
        /*
        if(i == dimension - 1)
        {
            gradient = MatrixMath::unitaryMatrix(dimensions[i+1], 1);
        }
        else
        {
            gradient = MatrixMath::dsigmoid(dimensions[i+1], 1, layers[i+1]);
        }
         */


        MatrixMath::Hadamard(dimensions[i+1], 1, gradient, errors[i]);

        MatrixMath::scalarMultiply(dimensions[i+1], 1, learningRate, gradient);

        float** layerT = MatrixMath::transpose(layers[i], dimensions[i], 1);

        deltaWeights[i] = MatrixMath::multiply(dimensions[i+1], 1, 1, dimensions[i], gradient, layerT);

        MatrixMath::freeMatrix(dimensions[i+1], 1, gradient);
        MatrixMath::freeMatrix(1, dimensions[i], layerT);

    }


    for(int i = 0; i < dimension; i++)
    {
        //update Weights
        MatrixMath::sum(dimensions[i+1], dimensions[i], weights[i], deltaWeights[i]);

        //update Biases
        MatrixMath::scalarMultiply(dimensions[i+1], 1, learningRate, errors[i]);
        MatrixMath::sum(dimensions[i+1], 1, biases[i+1], errors[i]);

        MatrixMath::freeMatrix(dimensions[i+1], dimensions[i], deltaWeights[i]);
        MatrixMath::freeMatrix(dimensions[i+1], 1, errors[i]);
    }

    delete errors;
    delete deltaWeights;
    //printBrain();
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
    auto* son = new BrainFart(father->dimensions, father->learningRate);

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
        for(int j = 0; j < dimensions[i+1]; j++)
        {
            delete weights[i][j];
        }
        delete weights[i];
    }
    delete weights;

    for(int i = 0; i < layerNumber; i++)
    {
        for(int j = 0; j < dimensions[i]; j++)
        {
            delete biases[i][j];
        }
        delete biases[i];
    }
    delete biases;
}

BrainFart *BrainFart::cloneBrain(BrainFart *copy) {
    auto* son = new BrainFart(copy->dimensions, copy->learningRate);


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

void BrainFart::printBrain()
{
    printf("Current Brain Weights\n");
    for(int i = 0; i < layerNumber-1; i++)
    {
        MatrixMath::print(weights[i], dimensions[i+1], dimensions[i]);
    }
}

void BrainFart::train(const TrainingStruct& input)
{
    std::vector<float> output = this->feedForward(input.Data);

    this->backwardPropagation(input.answer, output);

    this->freeLayers();
}




