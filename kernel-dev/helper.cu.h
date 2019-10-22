#ifndef HELPER
#define HELPER

#include <stdlib.h>
#include <stdio.h>

// *****************************************************************************
// Printing functions for validation
// *****************************************************************************

void printM(float* M, uint m, uint K){
    printf("[");
    for (int pix = 0; pix < m; pix++) {
        printf("[");
        for (size_t i = 0; i < K; i++){
            printf("[");
            for (size_t j = 0; j < K; j++){
                uint index = pix*K*K + i*K + j;
                printf(" %f", M[index]);
                if(j<K-1){printf(",");}
            }
            printf("]");
            if(i<K-1){printf(",");}
        }
        printf("]");
        if(pix<m-1){printf(",");}
    }
    printf("]");
}

void printVf(float* V, uint m, uint K){
    printf("[");
    for (uint i = 0; i < m; i++){
        printf("[");
        for (int j = 0; j < K; j++) {
            uint index = i*K + j;
            printf(" %f", V[index]);
            if(j<K-1){printf(",");}
        }
        printf("]");
        if(i<m-1){printf(",");}
    }
    printf("]");
}

void printVi(int* V, uint m, uint K){
    printf("[");
    for (uint i = 0; i < m; i++){
        printf("[");
        for (int j = 0; j < K; j++) {
            uint index = i*K + j;
            printf(" %d", V[index]);
            if(j<K-1){printf(",");}
        }
        printf("]");
        if(i<m-1){printf(",");}
    }
    printf("]");
}

void printE(uint* E, uint m){
    printf("[");
    for (uint i = 0; i < m; i++){
        printf(" %u", E[i]);
        if(i<m-1){printf(",");}
    }
    printf("]");
}

void printEi(int* E, uint m){
    printf("[");
    for (uint i = 0; i < m; i++){
        printf(" %d", E[i]);
        if(i<m-1){printf(",");}
    }
    printf("]");
}

void printEf(float* E, uint m){
    printf("[");
    for (uint i = 0; i < m; i++){
        printf(" %f", E[i]);
        if(i<m-1){printf(",");}
    }
    printf("]");
}


#endif

