#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

#include "helper.cu.h"

#define PI 3.14159265
#define F32_MIN -FLT_MAX
#define I32_MIN -2147483648
typedef unsigned int uint;

// M is the number of pixels
// K is the width and height of each of the matrix in A
// A is the input data [M][K][K]
// Ash is the [K][2*K]
// AI is the inverse A [K][K]
void gaussJordanG(uint M, uint K, float* A, float* AI){

    for (uint i = 0; i < M; i++){
        float* Ash    = (float*) calloc(2*K*K,sizeof(float));
        // Pad A with identity matrix to the right
        for (uint k1 = 0; k1 < K; k1++){
            for (uint k2 = 0; k2 < 2*K; k2++){
                if (k2<K) {
                    Ash[k1*2*K + k2] = A[i*K*2*K + k1*2*K + k2];
                } else {
                    Ash[k1*2*K + k2] = (float) (k2 == (K+k1));
                }
                // barrier
            }
        }

        // Gauss-Jordan Elimination:
        for (uint q = 0; q < K; q++){               // sequential
            float vq = Ash[0+q];
            for (uint k1 = 0; k1 < K; k1++){        // parallel block.y
                for (uint k2 = 0; k2 < 2*K; k2++){  // parallel block.x
                    float tmp = 0.0;
                    if (vq == 0.0) {
                        tmp = Ash[k1*2*K + k2];
                    } else {
                        float x = Ash[0+k2] / vq;
                        if (k1 == (K-1)){
                            tmp = x;
                        } else {
                            tmp = (Ash[(k1+1)*2*K + k2] - Ash[(k1+1)*2*K + q]) * x;
                        }
                    }
                    // barrier
                    Ash[k1*2*K + k2] = tmp;
                    // barrier
                }
            }
        }

        // collective copy shared-to-global mem:
        for (int k1 = 0; k1 < K; k1++) {
            for (int j = 0; j < K; j++) {
                uint XinvIdx  = k1*(K*2) + j+K;
                uint XlessIdx = i*K*K + k1*K + j;
                AI[XlessIdx] = Ash[XinvIdx];
            }
        }
        free(Ash);
    }
}
