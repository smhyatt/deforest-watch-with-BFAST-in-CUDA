#include <stdio.h>
#include <stdlib.h>
typedef unsigned int uint;
#define R 3
#define C 6

// integer i, j, k, n; real sum, xmult
// real array (ai j )1:n×1:n , (bi )1:n , (xi )1:n
// for k = 1 to n − 1 do           // row
//     for i = k + 1 to n do       // row we are working on
//         xmult ← aik/akk
//         aik ← xmult
//         for j = k + 1 to n do   // column
//             aij ← aij −(xmult)akj
//         end for
//         bi ← bi − (xmult)bk
//     end for
// end for
// xn ←bn/ann
// for i =n−1to1 step−1do
//     sum ← bi
//     for j = i + 1 to n do
//         sum ← sum − ai j x j
//     end for
// xi ← sum/aii
// end for
// end procedure Naive Gauss

void gaussJordan2(float* _X, uint cols, uint K, float* _Xres){
    // Making the upper triangle
    // row = 0
    for (uint row = 0; row < K-1; row++){
        // rowWork = 2
        for (uint rowWork = row + 1; rowWork < K; rowWork++){
            // xMult = _X[2*6+0] / _X[0*6+0] = 1 / 1 = 1
            float xMult = _X[rowWork*cols+row] / _X[row*cols+row];
            // _Xres[2*6+0] = 1;
            // _Xres[rowWork*cols+row] = xMult;
            // col = 5
            for (uint col = row; col < cols; col++){
                // factorIdx = 0 * 6 + 5 = 5
                uint factorIdx   = row     * cols + col;
                // elemIdx = 2 * 6 + 5 = 17
                uint elemIdx     = rowWork * cols + col;
                // _Xres[5] = _X[17] - 1 * _X[5];
                // _Xres[5] = 1 - 1 * 0;
                _X[elemIdx] = _X[elemIdx] - xMult * _X[factorIdx];
            }
        }
    }

}

void printMatrix(float* X, uint numRows, uint numCols){
    for (uint i = 0; i < numRows; i++){
        printf("[");
        for (uint j = 0; j < numCols-1; j++){
            printf("%f, ",X[i*numCols+j]);
        }
        printf("%f, ",X[i*numCols+numRows-1]);
        printf("]\n");
    }
}

int main() {
    float X[R][C]   = {{1, 0, 1, 1, 0, 0}
                      ,{0, 2, 1, 0, 1, 0}
                      ,{1, 0, 1, 0, 0, 1}};

    float* Xinv     = calloc(R*C,sizeof(float));

    float XinvTrueFact[R][C]
                    = {{1, 0, 0,-1,-1, 2}
                      ,{0, 1, 0,-1, 0, 1}
                      ,{0, 0, 1, 2, 1,-2}};

    printf("Matrix X:\n");
    printMatrix(&X[0][0],R,C);
    gaussJordan2(&X[0][0],C,R,Xinv);
    printf("Matrix X:\n");
    printMatrix(&X[0][0],R,C);

    // printf("Matrix Xinv:\n");
    // printMatrix(Xinv,R,C);
    printf("Expected matrix XinvTrueFact:\n");
    printMatrix(&XinvTrueFact[0][0],R,C);
    return 0;
}
