#include "MatrixLib/MatrixLib.h"
#include "BrainLib/BrainFart.h"
#include <math.h>
#include<time.h>
#include<stdlib.h>


void printArr(float* arr, int size)
{
    for (int i = 0; i < size; i++) {
        printf("%f ", arr[i]);
    }
    printf("\n");
}

int main()
{
    srand(time(NULL));
    BrainFart* brain1 = new BrainFart({1, 3, 3, 1});

    for(int i = 0; i < 10; i++)
    {
        int input = rand()%50;
        printf("guessing %d squared\n", input);
        float *output = brain1->feedForward({float(input)});

        printArr(output, 1);

        std::vector<float> guess;
        for(int j = 0; j < 1; j++)
        {
            guess.push_back(output[j]);
        }

        brain1->backwardPropagation({float(input*input)}, guess);
    }

    brain1->freeLayers();

    brain1->freeBrain();
    return 0;
}