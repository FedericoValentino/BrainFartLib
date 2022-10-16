#include<cstdio>
#include<cstdlib>
#include<cstring>

void print(int** matrix, int r, int c)
{
    for(int i = 0; i < r; i++)
    {
        for(int j = 0; j < c; j++)
        {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

int** multiply(int r1, int c1, int r2, int c2, int** m1, int** m2)
{
    if(c1 != r2)
    {
        printf("Cannot Multiply!");
        return nullptr;
    }

    int** multiplication = new int*[r1];
    for(int i = 0; i < r2; i++)
    {
        multiplication[i] = new int[c2];
    }

    for(int r = 0; r < r1; r++)
    {
        for(int c = 0; c < c2; c++)
        {
            int sum = 0;
            for(int k = 0; k < c1; k++)
            {
                sum += m1[r][k] * m2[k][c];
            }
            multiplication[r][c] = sum;
        }
    }

    print(multiplication, r1, c2);
    return multiplication;
}


