#include <vector>
#include <list>

//TODO Backward Propagation

struct trainingStruct
{
    std::vector<float> Data;
    std::vector<float> answer;
};

typedef struct trainingStruct TrainingStruct;

class BrainFart{
public:
    BrainFart(std::vector<int> layerSizes, float LR);

    std::vector<float> feedForward(std::vector<float> input);

    void backwardPropagation(const std::vector<float>& actual, const std::vector<float>& guess);

    void train(const TrainingStruct& input);

    void printBrain();

    void mutate();

    void freeBrain();

    void freeLayers();

    static BrainFart* reproduce(BrainFart* father, BrainFart* mother);

    static BrainFart* cloneBrain(BrainFart* copy);

    int layerNumber;

private:
    float learningRate;

    float reLU(float x);

    float sig(float x);

    std::vector<int> dimensions;

    float*** layers;

    float*** biases;

    float*** weights;

    void initializeWeightsAndBiases();
};


