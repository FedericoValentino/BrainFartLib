#include "MatrixLib/MatrixLib.h"
#include "BrainLib/BrainFart.h"


int main()
{
    BrainFart* brain1 = new BrainFart({2, 3, 2});

    for(int i = 0; i < 100; i++)
    {
        float *output = brain1->feedForward({1, 1});

        std::vector<float> outputVec;

        for (int j = 0; j < 2; j++) {
            printf("%f ", output[j]);
            outputVec.push_back(output[j]);
        }

        printf("\n");
        brain1->backwardPropagation({0, 1}, outputVec);

        brain1->freeLayers();
    }

    brain1->freeBrain();
    return 0;
}