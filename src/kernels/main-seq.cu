#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <time.h>

#include "helper.cu.h"
#include "sequential.cu.h"

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

void readNum(FILE *fp, char* buff) {
  int i = 0;
  char c = getc(fp);
  while( c != ',' && c != EOF) {
    buff[i++] = c;
    c = getc(fp); 
  }
  buff[i] = '\0';
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

    FILE *fp, *fpim, *fpV;
    uint m = 0;

    // opening files with input and fpV for validating the results
    if (argv[1][0] == '2') {
        if (argv[1][1] == 's') {
            fp   = fopen("../data/sahara2C.in", "r");
            fpim = fopen("../data/sahara2Cimages.in", "r");
            fpV  = fopen("../data/sahara2val.data","a+");
            m = 2;
        } else {
            fp   = fopen("../data/peru2C.in", "r");
            fpim = fopen("../data/peru2Cimages.in", "r");
            fpV  = fopen("../data/peru2val.data","a+");
            m = 2;
        }
    } else {
        if (argv[1][0] == 's') {
            fp   = fopen("../data/saharaC.in", "r");
            fpim = fopen("../data/saharaCimages.in", "r");
            fpV  = fopen("../data/saharaval.data","a+");
        } else {
            fp   = fopen("../data/peruC.in", "r");
            fpim = fopen("../data/peruCimages.in", "r");
            fpV  = fopen("../data/peruval.data","a+");
        }
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
    if (m == 0) {
        m = (uint)atoi(input7);
    }

    int   trend = atoi(input1);
    float freq  = atof(input4);
    float hfrac = atof(input5);
    float lam   = atof(input6);
    
    int K  = 2*k + 2;
    int mappingLen, imageLen, i = 0;

    // getting the lengths of mappingindices and images
    while (getc(fp) != EOF) { mappingLen++; }
    while (getc(fpim) != EOF) { imageLen++; }

    // rewinding the pointer to extract the data
    rewind(fpim);

    // creating each array
    char mappings[mappingLen];

    // scanning mappingindices
    fscanf(fpim, " %[^\n] ", mappings);

    // allocating host memory for mappingindices and pixels
    int* h_mappingindices = (int*) calloc(N,sizeof(int));
    float* h_Y = (float*) calloc(N*m,sizeof(float));    

    char buff[1000];
    for (int i = 0; i < N*m; i++) {
      readNum(fpim, buff);
      h_Y[i] = atof(buff);
    }

    // converting mappingindices from char* to int*
    char delim[] = ",";
    char *mapPtr = strtok(mappings, delim);


    // inserting data to mappingindices
    while(mapPtr != NULL) {
        h_mappingindices[i] = atoi(mapPtr);
        i++;
        mapPtr = strtok(NULL, delim);
    }


    // closing file with data
   fclose(fp);

   // Tile size for register tiling 
   const uint R = 30;

    // allocate host memory for X
    float* h_seq_X         = (float*) calloc(N*K,sizeof(float));
    float* h_seq_XT        = (float*) calloc(N*K,sizeof(float));
    float* h_seq_Xsqr      = (float*) calloc(K*K*m,sizeof(float));
    float* h_seq_Xinv      = (float*) calloc(K*K*m,sizeof(float));
    float* h_seq_B0        = (float*) calloc(K*m,sizeof(float));
    float* h_seq_B         = (float*) calloc(K*m,sizeof(float));
    float* h_seq_yhat      = (float*) calloc(N*m,sizeof(float));
    uint * h_seq_Nss       = (uint *) calloc(m  ,sizeof(uint));
    int  * h_seq_indss     = (int  *) calloc(m*N,sizeof(int));
    float* h_seq_yerrs     = (float*) calloc(N*m,sizeof(float));
    float* h_seq_yerrs_all = (float*) calloc(N*m,sizeof(float));
    uint * h_seq_nss       = (uint *) calloc(m  ,sizeof(uint));
    int  * h_seq_hs        = (int  *) calloc(m  ,sizeof(int));
    float* h_seq_sigmas    = (float*) calloc(m  ,sizeof(float));
    float* h_seq_MOfsts    = (float*) calloc(m  ,sizeof(float));
    float* h_seq_bound     = (float*) calloc(N-n,sizeof(float));
    float* h_seq_MOp       = (float*) calloc(m*(N-n),sizeof(float));
    float* h_seq_means     = (float*) calloc(m,sizeof(float));
    int*   h_seq_fstBreakP = (int  *) calloc(m,sizeof(int));
    float* h_seq_MOpp      = (float*) calloc(m*(N-n),sizeof(float));

    for (int i = 0; i < m*N; i++) { h_seq_yerrs[i] = F32_MIN; }

    /////////////////////////////////////////////////////////////////////////
    //// KERNEL 1
    /////////////////////////////////////////////////////////////////////////
    {
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        // calling sequential kernel 1 and transpose from the sequential file
        ker1seq(N, K, freq, h_mappingindices, h_seq_X);
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
        ker2seqtiled(n, N, m, h_seq_X, h_seq_XT, h_Y, h_seq_Xsqr, K, R);

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
        ker3seq(m, K, h_seq_Xsqr, h_seq_Xinv);

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
        printf("Sequential kernel 3 version runs in: %lu microsecs\n", elapsed);

        // validation
        printM(fpV, h_seq_Xinv, m, K);
    }

    /////////////////////////////////////////////////////////////////////////
    //// KERNEL 4
    /////////////////////////////////////////////////////////////////////////
    {
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        // calling sequential kernel 4
        ker4seqtiled(m, n, N, h_seq_X, K, h_Y, h_seq_B0);

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
        printf("Sequential kernel 4 version runs in: %lu microsecs\n", elapsed);

        // validation
        printVf(fpV, h_seq_B0, m, K);
    }


    /////////////////////////////////////////////////////////////////////////
    //// KERNEL 5
    /////////////////////////////////////////////////////////////////////////
    {
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        // calling sequential kernel 5
        ker5seqtiled(m, h_seq_Xinv, K, h_seq_B0, h_seq_B);

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
        printf("Sequential kernel 5 version runs in: %lu microsecs\n", elapsed);

        // validation
        printVf(fpV, h_seq_B, m, K);
    }

    /////////////////////////////////////////////////////////////////////////
    //// KERNEL 6
    /////////////////////////////////////////////////////////////////////////
    {
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        // calling sequential kernel 6
        // ker6seqnaive(m, N, h_seq_XT, h_seq_B, K, h_seq_yhat);
        ker6seqtiled(m, N, h_seq_XT, h_seq_B, K, h_seq_yhat);

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
        printf("Sequential kernel 6 version runs in: %lu microsecs\n", elapsed);

        // validation
        printVf(fpV, h_seq_yhat, m, N);
    }


    /////////////////////////////////////////////////////////////////////////
    //// KERNEL 7
    /////////////////////////////////////////////////////////////////////////
    {
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        // calling sequential kernel 7
        ker7seq(m, N, h_seq_yhat, h_seq_yerrs_all, h_seq_Nss, 
                                h_seq_yerrs, h_Y, h_seq_indss);

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
        printf("Sequential kernel 7 version runs in: %lu microsecs\n", elapsed);

        // validation
        printE(fpV,  h_seq_Nss, m);
        printVfnan(fpV, h_seq_yerrs, m, N);
        printVi(fpV, h_seq_indss, m, N);
    }


    /////////////////////////////////////////////////////////////////////////
    //// KERNEL 8
    /////////////////////////////////////////////////////////////////////////
    {
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        ker8seq(m, n, N, K, hfrac, h_seq_yerrs, h_Y,
                h_seq_nss, h_seq_hs, h_seq_sigmas);

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
        printf("Sequential kernel 8 version runs in: %lu microsecs\n", elapsed);

        // validation
        printE(fpV,  h_seq_nss, m);
        printEi(fpV, h_seq_hs,  m);
        printEf(fpV, h_seq_sigmas,  m);
    }


    /////////////////////////////////////////////////////////////////////////
    //// KERNEL 9
    /////////////////////////////////////////////////////////////////////////
    {
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        // calling sequential kernel 9
        ker9seq(m, N, h_seq_hs, h_seq_yerrs, h_seq_nss, h_seq_MOfsts);

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
        printf("Sequential kernel 9 version runs in: %lu microsecs\n", elapsed);
        
        // validation
        printEf(fpV, h_seq_MOfsts, m);
    }

    /////////////////////////////////////////////////////////////////////////
    //// KERNEL 10
    /////////////////////////////////////////////////////////////////////////
    {
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        // calling sequential kernel 10
        ker10seq(lam, m, n, N, h_seq_bound, h_seq_Nss,
                h_seq_nss,
                h_seq_sigmas,
                h_seq_hs,
                h_mappingindices,
                h_seq_MOfsts,
                h_seq_yerrs,
                h_seq_indss,
                h_seq_MOp,
                h_seq_means,
                h_seq_fstBreakP,
                h_seq_MOpp);

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
        printf("Sequential kernel 10 version runs in: %lu microsecs\n", elapsed);
        
        // validation
        printEi(fpV, h_seq_fstBreakP, m);
        printEf(fpV, h_seq_means, m);
    }

    fclose(fpV);

    // 7. clean up memory
    free(h_mappingindices);
    free(h_Y);
    free(h_seq_X);
    free(h_seq_XT);
    free(h_seq_Xsqr);
    free(h_seq_Xinv);
    free(h_seq_B0);
    free(h_seq_B);
    free(h_seq_yhat);
    free(h_seq_bound);
    free(h_seq_MOp);
    free(h_seq_means);
    free(h_seq_fstBreakP);
    free(h_seq_MOpp);
    free(h_seq_MOfsts);
    free(h_seq_indss);
    free(h_seq_nss);
    free(h_seq_Nss);
    free(h_seq_sigmas);
    free(h_seq_hs);
    free(h_seq_yerrs);
    free(h_seq_yerrs_all);
}







