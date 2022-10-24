#include "MatrixLib.h"

void MatrixMath::print(float** matrix, int r, int c)
{

    printf("Matrix %d x %d\n", r, c);
    for(int i = 0; i < r; i++)
    {
        for(int j = 0; j < c; j++)
        {
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
}

float** MatrixMath::multiply(int r1, int c1, int r2, int c2, float** m1, float** m2)
{
    if(c1 != r2)
    {
        printf("Cannot Multiply!");
        return nullptr;
    }



    auto** multiplication = new float*[r1];

    for(int i = 0; i < r1; i++)
    {
        multiplication[i] = new float[c2];
    }



    for(int r = 0; r < r1; r++)
    {
        for(int c = 0; c < c2; c++)
        {
            multiplication[r][c] = 0;
            for(int k = 0; k < c1; k++)
            {
                multiplication[r][c] += m1[r][k] * m2[k][c];
            }
        }
    }
    return multiplication;
}

float** MatrixMath::transpose(float** m1, int r, int c)
{
    float** transpose = new float*[c];
    for(int i = 0; i < c; i++)
    {
        transpose[i] = new float[r];
    }

    for(int i = 0; i < c; i++)
    {
        for(int j = 0; j < r; j++)
        {
            transpose[i][j] = m1[j][i];
        }
    }

    return transpose;
}

float **MatrixMath::subtract(int r1, int c1, float **m1, float **m2) {
    float** returnMatrix = new float*[r1];
    for(int i = 0; i < c1; i++)
    {
        returnMatrix[i] = new float[c1];
    }

    for(int i = 0; i < r1; i++)
    {
        for(int j = 0; j < c1; j++)
        {
            returnMatrix[i][j] = m1[i][j] - m2[i][j];
        }
    }

    return returnMatrix;
}

float **MatrixMath::toMatrix(int r, int c, std::vector<float> input) {
    float** toMatrix = new float*[r];
    for(int i = 0; i < r; i++)
    {
        toMatrix[i] = new float[c];
    }

    for(int i = 0; i < r; i++)
    {
        for(int j = 0; j < c; j++)
        {
            toMatrix[i][j] = input[i*r + j];
        }
    }

    return toMatrix;
}

void MatrixMath::Hadamard(int r, int c, float **m1, float **m2)
{
    for(int i = 0; i < r; i++)
    {
        for(int j = 0; j < c; j++)
        {
            m1[i][j] = m1[i][j] * (m2[i][j] * (1 - m2[i][j]));
        }
    }
}

void MatrixMath::sum(int r, int c, float **m1, float **m2)
{
    for(int i = 0; i < r; i++)
    {
        for(int j = 0; j < c; j++)
        {
            m1[i][j] = m1[i][j] + m2[i][j];
        }
    }

}

