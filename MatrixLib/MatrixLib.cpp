#include<cstdio>

void print(float** matrix, int r, int c)
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

float** multiply(int r1, int c1, int r2, int c2, float** m1, float** m2)
{
    if(c1 != r2)
    {
        printf("Cannot Multiply!");
        return nullptr;
    }

    float** multiplication = new float*[r1];
    for(int i = 0; i < r2; i++)
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


