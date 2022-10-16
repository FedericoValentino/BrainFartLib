#include <vector>
#include <list>

class BrainFart{
public:
    BrainFart(std::vector<int> layerSizes);

private:
    float reLU(float x);

    std::vector<int> dimensions;

    float** layers;

    float*** weights;

    void initializeWeights();
};


