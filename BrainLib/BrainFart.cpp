#include "BrainFart.h"

float BrainFart::reLU(float x) {
    if(x > 0)
    {
        return x;
    }
    return 0;
}

BrainFart::BrainFart(std::vector<int> layerSizes)
{

    layers = new float*[layerSizes.size()];
    for(int i = 0; i < layerSizes.size(); i++)
    {
        layers[i] = new float[layerSizes[i]];
    }

    weights = new float ** [layerSizes.size()-1];

    for(int i = 0; i < layerSizes.size()-2; i++)
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

}
