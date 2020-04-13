#ifndef HELPER
#define HELPER

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define F32_MIN -FLT_MAX

// *****************************************************************************
// Printing functions for validation
// *****************************************************************************
void printX(FILE* fp, float* M, uint K, uint N){
    fprintf(fp, "[");
    for (size_t i = 0; i < K; i++){
        fprintf(fp, "[");
        for (size_t j = 0; j < N; j++){
            uint index = i*N + j;
            fprintf(fp, " %ff32", M[index]);
            if(j<N-1){fprintf(fp, ",");}
        }
        fprintf(fp, "]");
        if(i<K-1){fprintf(fp, ",");}
    }
    fprintf(fp, "]");
}

void printM(FILE* fp, float* M, uint m, uint K){
    fprintf(fp, "[");
    for (int pix = 0; pix < m; pix++) {
        fprintf(fp, "[");
        for (size_t i = 0; i < K; i++){
            fprintf(fp, "[");
            for (size_t j = 0; j < K; j++){
                uint index = pix*K*K + i*K + j;
                fprintf(fp, " %f", M[index]);
                if(j<K-1){fprintf(fp, ",");}
            }
            fprintf(fp, "]");
            if(i<K-1){fprintf(fp, ",");}
        }
        fprintf(fp, "]");
        if(pix<m-1){fprintf(fp, ",");}
    }
    fprintf(fp, "]");
}

void printVf(FILE* fp, float* V, uint m, uint K){
    fprintf(fp, "[");
    for (uint i = 0; i < m; i++){
        fprintf(fp, "[");
        for (int j = 0; j < K; j++) {
            uint index = i*K + j;
            fprintf(fp, " %f", V[index]);
            if(j<K-1){fprintf(fp, ",");}
        }
        fprintf(fp, "]");
        if(i<m-1){fprintf(fp, ",");}
    }
    fprintf(fp, "]");
}

// special case for kernel 7, y_errors
void printVfnan(FILE* fp, float* V, uint m, uint K){
    fprintf(fp, "[");
    for (uint i = 0; i < m; i++){
        fprintf(fp, "[");
        for (int j = 0; j < K; j++) {
            uint index = i*K + j;
            float tmpV = V[index];
            if (tmpV == F32_MIN) {
                fprintf(fp, " f32.nan");
            } else {
                fprintf(fp, " %f", tmpV);
            }
            if(j<K-1){fprintf(fp, ",");}
        }
        fprintf(fp, "]");
        if(i<m-1){fprintf(fp, ",");}
    }
    fprintf(fp, "]");
}


void printVi(FILE* fp, int* V, uint m, uint K){
    fprintf(fp, "[");
    for (uint i = 0; i < m; i++){
        fprintf(fp, "[");
        for (int j = 0; j < K; j++) {
            uint index = i*K + j;
            fprintf(fp, " %d", V[index]);
            if(j<K-1){fprintf(fp, ",");}
        }
        fprintf(fp, "]");
        if(i<m-1){fprintf(fp, ",");}
    }
    fprintf(fp, "]");
}

void printE(FILE* fp, uint* E, uint m){
    fprintf(fp, "[");
    for (uint i = 0; i < m; i++){
        fprintf(fp, " %u", E[i]);
        if(i<m-1){fprintf(fp, ",");}
    }
    fprintf(fp, "]");
}

void printEi(FILE* fp, int* E, uint m){
    fprintf(fp, "[");
    for (uint i = 0; i < m; i++){
        fprintf(fp, " %d", E[i]);
        if(i<m-1){fprintf(fp, ",");}
    }
    fprintf(fp, "]");
}

void printEf(FILE* fp, float* E, uint m){
    fprintf(fp, "[");
    for (uint i = 0; i < m; i++){
        fprintf(fp, " %f", E[i]);
        if(i<m-1){fprintf(fp, ",");}
    }
    fprintf(fp, "]");
}


#endif

