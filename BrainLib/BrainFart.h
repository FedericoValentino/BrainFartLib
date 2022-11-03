#include <vector>
#include <list>

//TODO Backward Propagation

class BrainFart{
public:
    BrainFart(std::vector<int> layerSizes);

    float* feedForward(std::vector<float> input);

    void backwardPropagation(const std::vector<float>& actual, const std::vector<float>& guess);

    void printBrain();

    void mutate();

    void freeBrain();

    void freeLayers();

    static BrainFart* reproduce(BrainFart* father, BrainFart* mother);

    static BrainFart* cloneBrain(BrainFart* copy);

    int layerNumber;

private:
    float reLU(float x);

    float sig(float x);

    std::vector<int> dimensions;

    float*** layers;

    float*** biases;

    float*** weights;

    void initializeWeightsAndBiases();
};


