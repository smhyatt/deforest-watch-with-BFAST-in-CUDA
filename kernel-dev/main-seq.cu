#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <time.h>

#include "helper.cu.h"
#include "sequential.cu.h"

#define BLOCK_SIZE 1024//1024 //1024//2048
#define WIDTH_A  1024//1024 //1024//2048
#define HEIGHT_A 1//2048//2048//2048
#define WIDTH_B  1024//4096//2048
#define TILE_HEIGHT 1
#define TILE_WIDTH 1024
#define F32_MIN -FLT_MAX
#define I32_MIN -2147483648
typedef unsigned int uint;


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//// Helpers
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1) {
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}


void randomInit(float* data, int size) {
   for (int i = 0; i < size; ++i)
   data[i] = rand() / (float)RAND_MAX;
}


template<class T>
void matMult(T* A, T* B, T* C, int colsA, int rowsA, int colsB) {
  for(int i = 0; i < rowsA; i++) {
    for(int j = 0; j < colsB; j++) {
      float sum = 0.0;
      for(int k = 0; k < colsA; k++) {
        sum += A[i*colsA + k] * B[k * colsB + j];
      }
      C[i * colsB + j] = sum;
    }
  }
}

template<class T>
bool validate(float* A,float* B, unsigned int sizeAB){
    for(int i = 0; i < sizeAB; i++)
      if (fabs(A[i] - B[i]) > 0.0005){
        printf("INVALID RESULT %d %f %f\n", i, A[i], B[i]);
        return false;
      }
    printf("VALID RESULT!\n");
    return true;
}


///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//// PROGRAM MAIN
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

int main(int argc, char const *argv[]) {
   if (argc != 2) {
      printf("Please include the name of the dataset.\n");
         return -1;
   }

///////////////////////////////////////////////////////////////////////////////
//// PARSING
///////////////////////////////////////////////////////////////////////////////

    FILE *fp, *fpim;

    if (argv[1][0] == 's') {
        fp   = fopen("../data/saharaC.in", "r");
        fpim = fopen("../data/saharaCimages.in", "r");
    } else {
        fp   = fopen("../data/peruC.in", "r");
        fpim = fopen("../data/peruCimages.in", "r");
    }

    if (fp == NULL || fpim == NULL) {
        printf("Files not read.\n");
        return -1;
    }

    char input1[10], input2[10], input3[30], input4[30];
    char input5[30], input6[30], input7[50], input8[30];
    fscanf(fp, " %[^\n]  %[^\n]  %[^\n]  %[^\n] ", input1,input2,input3,input4);
    fscanf(fp, " %[^\n]  %[^\n]  %[^\n]  %[^\n] ", input5,input6,input7,input8);
    int  k    = atoi(input2);
    uint n    = (uint)atoi(input3);
    uint N    = (uint)atoi(input8);
    uint mIRL = (uint)atoi(input7);
    int trend = atoi(input1);
    float freq  = atof(input4);
    float hfrac = atof(input5);
    float lam   = atof(input6);
    uint m = 2;
    int K  = 2*k + 2;

    int mappingLen, imageLen, i = 0;

    // getting the lengths of mappingindices and images
    while (getc(fp) != EOF) { mappingLen++; }
    while (getc(fpim) != EOF) { imageLen++; }

    // rewinding the pointer to extract the data
    rewind(fpim);

    // extracting each array
    char mappings[mappingLen], pixels[(imageLen-mappingLen)];
    fscanf(fpim, " %[^\n]  %[^\n] ", mappings, pixels);

    // converting mappingindices from char* to int*
    char delim[] = ",";
    char *mapPtr = strtok(mappings, delim);

    // allocating host memory for mappingindices and pixels
    int* h_mappingindices = (int*) calloc(N,sizeof(int));
    float* h_sample = (float*) calloc(N*m,sizeof(float));

    // inserting data to mappingindices
    while(mapPtr != NULL) {
        h_mappingindices[i] = atoi(mapPtr);
        i++;
        mapPtr = strtok(NULL, delim);
    }

    // converting samples from char* to float*
    char *pixelsPtr = strtok(pixels, delim);
    i = 0;

    // inserting data to sample
    while(pixelsPtr != NULL) {
        h_sample[i] = atof(pixelsPtr);
        i++;
        pixelsPtr = strtok(NULL, delim);
    }

    // closing file with data
    fclose(fp);

    // opening file for validation of results
    FILE* fpV = fopen("../data/val.data","a+");

    // allocate host memory for X
    float* h_seq_X    = (float*) calloc(N*K,sizeof(float));
    float* h_seq_XT   = (float*) calloc(N*K,sizeof(float));
    float* h_seq_Xsqr = (float*) calloc(K*K*m,sizeof(float));
    float* h_seq_B0   = (float*) calloc(K*m,sizeof(float));

    /////////////////////////////////////////////////////////////////////////
    //// KERNEL 1
    /////////////////////////////////////////////////////////////////////////
    {
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        // calling sequential kernel 1 and transpose from the sequential file
        mkX(N, K, freq, h_mappingindices, h_seq_X);
        transpose(N, K, h_seq_X, h_seq_XT);

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
        printf("Sequential kernel 1 version runs in: %lu microsecs\n", elapsed);

        // validation 
        printX(fpV, h_seq_X, K, N);
    }

    /////////////////////////////////////////////////////////////////////////
    //// KERNEL 2
    /////////////////////////////////////////////////////////////////////////
    {
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        // calling sequential kernel 2
        mkXsqr(n, N, m, h_seq_X, h_seq_XT, h_sample, h_seq_Xsqr, K);

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
        printf("Sequential kernel 2 version runs in: %lu microsecs\n", elapsed);

        // validation 
        printM(fpV, h_seq_Xsqr, m, K);
    }

    /////////////////////////////////////////////////////////////////////////
    //// KERNEL 3
    /////////////////////////////////////////////////////////////////////////
    {
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        // calling sequential kernel 3
        matInv(m, h_seq_Xsqr, h_seq_XsqrInv, K);

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
        printf("Sequential kernel 3 version runs in: %lu microsecs\n", elapsed);
    }

    /////////////////////////////////////////////////////////////////////////
    //// KERNEL 4
    /////////////////////////////////////////////////////////////////////////
    {
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        // calling sequential kernel 4
        mkB0(m, n, N, h_seq_X, K, h_sample, h_seq_B0)

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
        printf("Sequential kernel 4 version runs in: %lu microsecs\n", elapsed);

        // validation 
        printVf(h_seq_B0, m, K);
    }

    /////////////////////////////////////////////////////////////////////////
    //// KERNEL 5
    /////////////////////////////////////////////////////////////////////////
    {
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        // calling sequential kernel 5

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
        printf("Sequential kernel 5 version runs in: %lu microsecs\n", elapsed);
    }

    /////////////////////////////////////////////////////////////////////////
    //// KERNEL 6
    /////////////////////////////////////////////////////////////////////////
    {
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        // calling sequential kernel 6

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
        printf("Sequential kernel 6 version runs in: %lu microsecs\n", elapsed);
    }


    /////////////////////////////////////////////////////////////////////////
    //// KERNEL 7
    /////////////////////////////////////////////////////////////////////////
    {
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        // calling sequential kernel 7

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
        printf("Sequential kernel 7 version runs in: %lu microsecs\n", elapsed);
    }

    /////////////////////////////////////////////////////////////////////////
    //// KERNEL 8
    /////////////////////////////////////////////////////////////////////////
    {
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        // calling sequential kernel 8

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
        printf("Sequential kernel 8 version runs in: %lu microsecs\n", elapsed);
    }


    /////////////////////////////////////////////////////////////////////////
    //// KERNEL 9
    /////////////////////////////////////////////////////////////////////////
    {
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        // calling sequential kernel 9

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
        printf("Sequential kernel 9 version runs in: %lu microsecs\n", elapsed);
    }

    /////////////////////////////////////////////////////////////////////////
    //// KERNEL 10
    /////////////////////////////////////////////////////////////////////////
    {
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        // calling sequential kernel 10

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
        printf("Sequential kernel 10 version runs in: %lu microsecs\n", elapsed);
    }


    fclose(fpV);

    // 7. clean up memory
    free(h_mappingindices);
    free(h_sample);
    free(h_seq_X);
    free(h_seq_XT);
}



