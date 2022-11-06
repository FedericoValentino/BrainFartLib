#include "MatrixLib/MatrixLib.h"
#include "BrainLib/BrainFart.h"
#include <math.h>
#include<time.h>
#include<stdlib.h>

/*
 * This is an example for a neural network capable of approximating the XOR function
 */



void printArr(std::vector<float> arr, int size)
{
    for (int i = 0; i < size; i++) {
        printf("%f ", arr[i]);
    }
    printf("\n");
}

int main()
{
    srand(time(NULL));
    BrainFart* brain1 = new BrainFart({2, 3, 1}, 0.06);

    for(int i = 0; i < 100000; i++)
    {
        int caso = rand()%4;
        TrainingStruct examples;
        switch(caso)
        {
            case 0:
                examples.Data = {1, 1};
                examples.answer = {0};
                break;
            case 1:
                examples.Data = {0, 0};
                examples.answer = {0};
                break;
            case 2:
                examples.Data = {1, 0};
                examples.answer = {1};
                break;
            case 3:
                examples.Data = {0, 1};
                examples.answer = {1};
                break;
        }

        brain1->train(examples);
    }


    printArr(brain1->feedForward({1, 1}), 1);
    brain1->freeLayers();
    printArr(brain1->feedForward({1, 0}), 1);
    brain1->freeLayers();
    printArr(brain1->feedForward({0, 1}), 1);
    brain1->freeLayers();
    printArr(brain1->feedForward({0, 0}), 1);
    brain1->freeLayers();

    brain1->freeBrain();

    delete brain1;
    return 0;
}