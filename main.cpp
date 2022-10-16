#include "MatrixLib/MatrixLib.h"
#include "BrainLib/BrainFart.h"


int main()
{
    BrainFart* brain = new BrainFart({9, 7, 7, 7, 9});

    brain->feedForward({1, 1, 1, 1, 1, 1, 1, 1, 1});

    return 0;
}