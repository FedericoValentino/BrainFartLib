#include "MatrixLib/MatrixLib.h"
#include "BrainLib/BrainFart.h"
#include <math.h>


int main()
{
    BrainFart* brain1 = new BrainFart({9, 18, 18, 9});

    float* output = brain1->feedForward({1, 1, -1,
                               1, -1, 0,
                              0, -1, 1});

    std::vector<float> guess;

    for(int i = 0; i < 9; i++)
    {
        guess.push_back(output[i]);
    }

    brain1->backwardPropagation({1, 1, -1, 1, -1, 0, 1, -1, 1},guess);


    brain1->freeLayers();


    brain1->freeBrain();
    return 0;
}