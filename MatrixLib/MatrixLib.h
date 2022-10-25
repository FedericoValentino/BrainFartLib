#include<cstdio>
#include <vector>

class MatrixMath{
public:
    static void print(float** matrix, int r, int c);

    static float** multiply(int r1, int c1, int r2, int c2, float** m1, float** m2);

    static float** transpose(float** m1, int r, int c);

    static void sum(int r, int c, float** m1, float** m2);

    static float** subtract(int r1, int c1, float** m1, float** m2);

    static float** toMatrix(int r, int c, std::vector<float> input);

    static void Hadamard(int r, int c, float** m1, float** m2);

    static void freeMatrix(int r, int c, float** m1);

    static float** dsigmoid(int r, int c, float** m1);


};

