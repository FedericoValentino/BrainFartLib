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
    BrainFart* brain1 = new BrainFart({2, 3, 1});

    for(int i = 0; i < 100000; i++)
    {
        int caso = rand()%4;
        std::vector<float> trainingData;
        std::vector<float> answer;
        switch(caso)
        {
            case 0:
                trainingData = {1, 1};
                answer = {0};
                break;
            case 1:
                trainingData = {0, 1};
                answer = {1};
                break;
            case 2:
                trainingData = {1, 0};
                answer = {1};
                break;
            case 3:
                trainingData = {0, 0};
                answer = {0};
                break;
        }

        float* output = brain1->feedForward(trainingData);

        std::vector<float> guess;

        guess.push_back(output[0]);

        brain1->backwardPropagation(answer, guess);

        brain1->freeLayers();
    }


    printArr(brain1->feedForward({1, 1}), 1);
    printArr(brain1->feedForward({1, 0}), 1);
    printArr(brain1->feedForward({0, 1}), 1);
    printArr(brain1->feedForward({0, 0}), 1);




    brain1->freeLayers();

    brain1->freeBrain();
    return 0;
}