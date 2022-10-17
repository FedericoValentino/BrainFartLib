#include "MatrixLib/MatrixLib.h"
#include "BrainLib/BrainFart.h"


int main()
{
    BrainFart* brain1 = new BrainFart({9, 7, 7, 7, 9});
    BrainFart* brain2 = new BrainFart({9, 7, 7, 7, 9});

    brain1->feedForward({1, 0, -1, 1, -1, 0, 0, 0, 0});

    BrainFart* son = BrainFart::reproduce(brain1, brain2);
    return 0;
}