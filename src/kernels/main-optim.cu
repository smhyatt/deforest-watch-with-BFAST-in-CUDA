#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <time.h>

#include "helper.cu.h"
#include "kernels-optim.cu.h"
#include "pbbKernels.cu.h"
#include "sequential.cu.h"

// Bit to set naive or optimized kernel runs 
#define NAIVE2 0
#define NAIVE3 0
#define NAIVE4 0
#define NAIVE5 0
#define NAIVE6 0
#define NAIVE8 0
#define F32_MIN -FLT_MAX
#define I32_MIN -2147483648
#define RUNS_GPU 100

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

int gpuAssert(cudaError_t code) {
    if(code != cudaSuccess) {
      printf("GPU Error: %s\n", cudaGetErrorString(code));
      return -1;
    }
    return 0;
}



// Cosmin's Matrix Transpose Wrapper from Weekly 3
void transposeTiled ( float*     inp_d,  
                      float*     out_d, 
                      const uint height, 
                      const uint width,
                      const uint T
) {
   // 1. setup block and grid parameters
   unsigned int sh_mem_size = T * (T+1) * sizeof(float); 
   int  dimy = (height+T-1) / T; 
   int  dimx = (width +T-1) / T;
   dim3 block(T, T, 1);
   dim3 grid (dimx, dimy, 1);

   //2. execute the kernel
   matTransposeTiledKer<<< grid, block, sh_mem_size >>> (inp_d, out_d, height, width, T); 
   // cudaThreadSynchronize();
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
    int Nmn = N-n;

    // getting the lengths of mappingindices and images
    while (getc(fp) != EOF)   { mappingLen++; }
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

   // allocate device memory
   uint map_size = N*sizeof(int);
   uint sam_size = N*m*sizeof(float);
   int* d_mappingindices;
   float* d_Y;
   cudaMalloc((void**) &d_mappingindices, map_size);
   cudaMalloc((void**) &d_Y, sam_size);


   // copy host memory to device
   cudaMemcpy(d_mappingindices, h_mappingindices, map_size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_Y, h_Y, sam_size, cudaMemcpyHostToDevice);

   uint X_size     = K*N*sizeof(float);
   uint Y_size     = N*m*sizeof(float);
   uint Nss_size   = N*m*sizeof(uint);
   uint nss_size   = m*sizeof(uint);
   uint hs_size    = m*sizeof(int);
   uint sigmas_size= m*sizeof(float);
   uint I_size     = N*m*sizeof(int);
   uint Xsqr_size  = K*K*m*sizeof(float);
   uint B0_size    = K*m*sizeof(float);
   uint MO_size    = m*sizeof(float);
   uint MOp_size   = Nmn*sizeof(float);
   uint MOpp_size  = Nmn*sizeof(float);
   uint means_size = m*sizeof(float);
   uint breaks_size= m*sizeof(int);
   uint bound_size = (N-n)*sizeof(float);

   // Tile size for register tiling 
   const uint R = 30;

   // allocate host memory
   float* h_X      = (float*) calloc(N*K,sizeof(float));
   float* h_XT     = (float*) calloc(K*N,sizeof(float));
   float* h_Xsqr   = (float*) calloc(K*K*m,sizeof(float));
   float* h_B0     = (float*) calloc(K*m,sizeof(float));
   float* h_B      = (float*) calloc(K*m,sizeof(float));
   float* h_yhat   = (float*) calloc(N*m,sizeof(float));
   float* h_Xinv   = (float*) calloc(K*K*m,sizeof(float));
   float* h_yerrs  = (float*) calloc(m*N,sizeof(float));
   uint * h_Nss    = (uint *) calloc(m*N,sizeof(uint));
   uint * h_nss    = (uint *) calloc(m,sizeof(uint));
   float* h_sigmas = (float*) calloc(m,sizeof(float));
   int  * h_indss  = (int  *) calloc(m*N,sizeof(int));
   int  * h_hs     = (int  *) calloc(m,sizeof(int));
   float* h_MOfsts = (float*) calloc(m,sizeof(float));
   float* h_bounds = (float*) calloc(N-n,sizeof(float));
   int*   h_breaks = (int  *) calloc(m,sizeof(int));
   float* h_means  = (float*) calloc(m,sizeof(float));

   // allocate device memory
   float *d_X, *d_XT, *d_Xsqr, *d_Xinv, *d_YT, *d_B0, *d_B, *d_yhat;
   float *d_yerrs, *d_MOfsts, *d_sigmas, *d_MOp, *d_means, *d_MOpp, *d_bounds;
   uint  *d_Nss,   *d_nss;
   int   *d_indss, *d_hs, *d_breaks;
   cudaMalloc((void**) &d_X, X_size);
   cudaMalloc((void**) &d_XT, X_size);
   cudaMalloc((void**) &d_YT, Y_size);
   cudaMalloc((void**) &d_Xsqr, Xsqr_size);
   cudaMalloc((void**) &d_Xinv, Xsqr_size);
   cudaMalloc((void**) &d_B0, B0_size);
   cudaMalloc((void**) &d_B, B0_size);
   cudaMalloc((void**) &d_yhat, Y_size);
   cudaMalloc((void**) &d_yerrs, Y_size);
   cudaMalloc((void**) &d_Nss, Nss_size);
   cudaMalloc((void**) &d_nss, nss_size);
   cudaMalloc((void**) &d_sigmas, sigmas_size);
   cudaMalloc((void**) &d_indss, I_size);
   cudaMalloc((void**) &d_hs, hs_size);
   cudaMalloc((void**) &d_MOfsts, MO_size);
   cudaMalloc((void**) &d_MOp, MOp_size);
   cudaMalloc((void**) &d_MOpp, MOpp_size);
   cudaMalloc((void**) &d_breaks, breaks_size);
   cudaMalloc((void**) &d_means, means_size);
   cudaMalloc((void**) &d_bounds, bound_size);



   /////////////////////////////////////////////////////////////////////////
   //// KERNEL 1
   /////////////////////////////////////////////////////////////////////////
   {
      dim3 block(256, 1, 1);
      dim3 grid ((N*K+256-1)/256, 1, 1);

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL);
      
      // Calling the sequential version of X
      ker1seq(N, K, freq, h_mappingindices, h_X);
      transpose(N, K, h_X, h_XT);
      cudaMemcpy(d_X, h_X, X_size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_XT, h_XT, X_size, cudaMemcpyHostToDevice);

      // GPU call to kernel 1
      // ker1 <<< grid, block >>>(N, K, freq, d_mappingindices, d_X, d_XT);
      // cudaDeviceSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

      // check for cuda errors
      gpuAssert( cudaPeekAtLastError() );

      // copy result from device to host
      // cudaMemcpy(h_X, d_X, X_size, cudaMemcpyDeviceToHost);
      // cudaMemcpy(h_XT, d_XT, X_size, cudaMemcpyDeviceToHost);

      // add to validation
      printX(fpV, h_X, K, N);

      printf("GPU Optimized Kernel 1 runs in: %lu microsecs\n", elapsed);
   }


   /////////////////////////////////////////////////////////////////////////
   //// KERNEL 2
   /////////////////////////////////////////////////////////////////////////
   {
      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL);

      #if (NAIVE2 == 1)
        dim3 block(K, K, 1);
        dim3 grid (m, 1, 1);
        for (int i = 0; i < RUNS_GPU; i++) {
          ker2naive<<< grid, block >>>(n, N, m, d_X, d_XT, d_Y, d_Xsqr, K);
        }
      #elif (NAIVE2 == 2)
        dim3 block(K, K, 1);
        dim3 grid ((m+R-1)/R, 1, 1);
        for (int i = 0; i < RUNS_GPU; i++) {
          transposeTiled(d_Y, d_YT, m, N, 32);
          ker2simpletiled<R><<< grid, block >>>(n, N, m, d_X, d_XT, d_YT, d_Xsqr, K);
        }
        cudaDeviceSynchronize();
      #else
        dim3 block(K, K, 1);
        dim3 grid ((m+R-1)/R, 1, 1);
        for (int i = 0; i < RUNS_GPU; i++) {
          transposeTiled(d_Y, d_YT, m, N, 32);
          ker2<R><<< grid, block >>> (n, N, m, d_X, d_XT, d_YT, d_Xsqr, K);
        }
        cudaDeviceSynchronize();
      #endif

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec)/RUNS_GPU;

      // check for cuda errors
      gpuAssert( cudaPeekAtLastError() );

      // copy result from device to host
      cudaMemcpy(h_Xsqr, d_Xsqr, Xsqr_size, cudaMemcpyDeviceToHost);

      // validation
      printM(fpV, h_Xsqr, m, K);

      printf("GPU Optimized Kernel 2 runs in: %lu microsecs\n", elapsed);
   }

   /////////////////////////////////////////////////////////////////////////
   //// KERNEL 3
   /////////////////////////////////////////////////////////////////////////
   {
      dim3 block(2*K, K, 1);
      dim3 grid (m, 1, 1);

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL);

      #if (NAIVE3 == 0)
        for (int i = 0; i < RUNS_GPU; i++) {
          ker3<<< grid, block, 4*K*K*sizeof(float) >>>(m, K, d_Xsqr, d_Xinv);
        }
        cudaDeviceSynchronize();
      #else
        for (int i = 0; i < RUNS_GPU; i++) {
          ker3seq(m, K, h_Xsqr, h_Xinv);
        }
        cudaMemcpy(d_Xinv, h_Xinv, Xsqr_size, cudaMemcpyHostToDevice);
      #endif
      
      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec)/RUNS_GPU;

      #if (NAIVE3 == 0)
        // check for cuda errors
        gpuAssert( cudaPeekAtLastError() );

        // copy result from device to host
        cudaMemcpy(h_Xinv, d_Xinv, Xsqr_size, cudaMemcpyDeviceToHost);
      #endif
      // validation
      printM(fpV, h_Xinv, m, K);
      // printM(fpVcosmin, h_Xinv, m, K);


      printf("GPU Optimized Kernel 3 runs in: %lu microsecs\n", elapsed);
   }


   /////////////////////////////////////////////////////////////////////////
   //// KERNEL 4
   /////////////////////////////////////////////////////////////////////////
   {
      dim3 block(K, K, 1);
      dim3 grid ((m+K-1)/K, 1, 1);

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL);

      #if NAIVE4
        for (int i = 0; i < RUNS_GPU; i++) {
          ker4simpletiled <<< grid, block >>> (m, n, N, d_X, K, d_Y, d_B0);
        }
        cudaDeviceSynchronize();
      #else
        for (int i = 0; i < RUNS_GPU; i++) {
          ker4 <<< grid, block >>> (m, n, N, d_X, K, d_Y, d_B0);
        }
        cudaDeviceSynchronize();
      #endif

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec)/RUNS_GPU;

      // check for cuda errors
      gpuAssert( cudaPeekAtLastError() );

      // copy result from device to host
      cudaMemcpy(h_B0, d_B0, B0_size, cudaMemcpyDeviceToHost);
      
      // add to validation
      printVf(fpV, h_B0, m, K);

      printf("GPU Optimized Kernel 4 runs in: %lu microsecs\n", elapsed);
   }

   /////////////////////////////////////////////////////////////////////////
   //// KERNEL 5
   /////////////////////////////////////////////////////////////////////////   
   {
      dim3 block(K, K, 1);
      dim3 grid ((m+K-1)/K, 1, 1);

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL);

      #if NAIVE5
        for (int i = 0; i < RUNS_GPU; i++) {
          ker5simpletiled <<< grid, block >>> (m, d_Xinv, K, d_B0, d_B);
        }
        cudaDeviceSynchronize();
      #else
        for (int i = 0; i < RUNS_GPU; i++) {
          ker5 <<< grid, block >>> (m, d_Xinv, K, d_B0, d_B);
        }
        cudaDeviceSynchronize();
      #endif

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec)/RUNS_GPU;

      // check for cuda errors
      gpuAssert( cudaPeekAtLastError() );

      // copy result from device to host
      cudaMemcpy(h_B, d_B, B0_size, cudaMemcpyDeviceToHost);
      
      // validation 
      printVf(fpV, h_B, m, K);

      printf("GPU Optimized Kernel 5 runs in: %lu microsecs\n", elapsed);
   }

   /////////////////////////////////////////////////////////////////////////
   //// KERNEL 6
   /////////////////////////////////////////////////////////////////////////
   {
      const int T = 16;
      int  dimx = (N+T-1)/T;
      int  dimy = (m+T-1)/T;
      dim3 block(T, T, 1);
      dim3 grid (dimx, dimy, 1);

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL);

      #if NAIVE6
        for (int i = 0; i < RUNS_GPU; i++) {
          ker6simpletiled<<< grid, block >>> (m, N, d_XT, d_B, K, d_yhat);
        }
        cudaDeviceSynchronize();
      #else
        for (int i = 0; i < RUNS_GPU; i++) {
          ker6<T> <<< grid, block >>> (m, N, d_XT, d_B, K, d_yhat);
        }
        cudaDeviceSynchronize();
      #endif

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec)/RUNS_GPU;

      // check for cuda errors
      gpuAssert( cudaPeekAtLastError() );

      // copy result from device to host
      cudaMemcpy(h_yhat, d_yhat, Y_size, cudaMemcpyDeviceToHost);

      // validation
      printVf(fpV, h_yhat, m, N);

      printf("GPU Optimized Kernel 6 runs in: %lu microsecs\n", elapsed);
   }

   /////////////////////////////////////////////////////////////////////////
   //// KERNEL 7
   /////////////////////////////////////////////////////////////////////////
   {
      dim3 block(N, 1, 1);
      dim3 grid (m, 1, 1);

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL);

      for (int i = 0; i < RUNS_GPU; i++) {
        ker7 <<< grid, block, 2*N*sizeof(float) >>> (m, N, d_yhat, 
                                      d_Nss, d_yerrs, d_Y, d_indss);
      }
      cudaDeviceSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec)/RUNS_GPU;

      // check for cuda errors
      gpuAssert( cudaPeekAtLastError() );

      // copy result from device to host
      cudaMemcpy(h_Nss,   d_Nss,   Nss_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_yerrs, d_yerrs, Y_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_indss, d_indss, I_size, cudaMemcpyDeviceToHost);

      // validation
      printE(fpV, h_Nss, m);
      printVfnan(fpV, h_yerrs, m, N);
      printVi(fpV, h_indss, m, N);

      printf("GPU Optimized Kernel 7 runs in: %lu microsecs\n", elapsed);
   }

   /////////////////////////////////////////////////////////////////////////
   //// KERNEL 8
   /////////////////////////////////////////////////////////////////////////
   {
      dim3 block(n, 1, 1);
      dim3 grid (m, 1, 1);

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL);

      #if NAIVE8
        for (int i = 0; i < RUNS_GPU; i++) {
          ker8naive <<< grid, block, n*sizeof(uint) + n*sizeof(float) >>>(m, n, N, K, 
                                      hfrac, d_yerrs, d_Y, d_nss, d_hs, d_sigmas);
        }
        cudaDeviceSynchronize();
      #else
        for (int i = 0; i < RUNS_GPU; i++) {
          ker8 <<< grid, block, n*sizeof(uint) + n*sizeof(float) >>>(m, n, N, K, 
                                      hfrac, d_yerrs, d_Y, d_nss, d_hs, d_sigmas);
        }
        cudaDeviceSynchronize();
      #endif

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec)/RUNS_GPU;

      // check for cuda errors
      gpuAssert( cudaPeekAtLastError() );

      // copy result from device to host
      cudaMemcpy(h_nss, d_nss, m*sizeof(uint), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_hs, d_hs, m*sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_sigmas, d_sigmas, m*sizeof(float), cudaMemcpyDeviceToHost);

      // validation
      printE(fpV,  h_nss, m);
      printEi(fpV, h_hs,  m);
      printEf(fpV, h_sigmas,  m);

      printf("GPU Optimized Kernel 8 runs in: %lu microsecs\n", elapsed);
    }


   /////////////////////////////////////////////////////////////////////////
   //// KERNEL 9
   /////////////////////////////////////////////////////////////////////////
   {
      dim3 block((n*hfrac), 1, 1);
      dim3 grid (m, 1, 1);

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL);

      for (int i = 0; i < RUNS_GPU; i++) {
        ker9 <<< grid, block, (n*hfrac)*sizeof(float) >>> (hfrac, n, m, N, 
                                        d_hs, d_yerrs, d_nss, d_MOfsts);
      }
      cudaDeviceSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec)/RUNS_GPU;

      // check for cuda errors
      gpuAssert( cudaPeekAtLastError() );

      // copy result from device to host
      cudaMemcpy(h_MOfsts, d_MOfsts, MO_size, cudaMemcpyDeviceToHost);

      // validation 
      printEf(fpV, h_MOfsts, m);

      printf("GPU Optimized Kernel 9 runs in: %lu microsecs\n", elapsed);
   }


    /////////////////////////////////////////////////////////////////////////
    //// KERNEL 10
    /////////////////////////////////////////////////////////////////////////
    {
        dim3 block(N-n, 1, 1);
        dim3 grid (m, 1, 1);
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        compBound(lam, n, N, Nmn, h_mappingindices, h_bounds);
        cudaMemcpy(d_bounds, h_bounds, bound_size, cudaMemcpyHostToDevice);

        for (int i = 0; i < RUNS_GPU; i++) {
          ker10 <<< grid, block, (N-n)*sizeof(float) >>> (lam, m, n, N, d_bounds,
                                d_Nss, d_nss, d_sigmas,  d_hs,
                                d_mappingindices, d_MOfsts,
                                d_yerrs, d_indss,  d_MOp,
                                d_means, d_breaks, d_MOpp);
        }
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec)/RUNS_GPU;

        // check for cuda errors
        gpuAssert( cudaPeekAtLastError() );

        // copy result from device to host
        cudaMemcpy(h_breaks, d_breaks, breaks_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_means, d_means, means_size, cudaMemcpyDeviceToHost);
        printEi(fpV, h_breaks, m);
        printEf(fpV, h_means, m);

        printf("GPU Optimized Kernel 10 runs in: %lu microsecs\n", elapsed);
    }

   
   fclose(fpV);



   // 7. clean up memory
   free(h_mappingindices);
   free(h_Y);
   free(h_X);
   free(h_XT);
   free(h_Xsqr);
   free(h_Xinv);
   free(h_B0);
   free(h_B);
   free(h_yhat);
   free(h_yerrs);
   free(h_Nss);
   free(h_nss);
   free(h_indss);
   free(h_hs);
   free(h_MOfsts);
   free(h_sigmas);
   free(h_bounds);
   free(h_breaks);
   free(h_means);
   cudaFree(d_X);
   cudaFree(d_XT);
   cudaFree(d_YT);
   cudaFree(d_Xsqr);
   cudaFree(d_Xinv);
   cudaFree(d_B0);
   cudaFree(d_B);
   cudaFree(d_yhat);
   cudaFree(d_yerrs);
   cudaFree(d_Nss);
   cudaFree(d_nss);
   cudaFree(d_breaks);
   cudaFree(d_indss);
   cudaFree(d_hs);
   cudaFree(d_MOfsts);
   cudaFree(d_bounds);
   cudaFree(d_sigmas);
   cudaFree(d_MOp);
   cudaFree(d_MOpp);
   cudaFree(d_means);
   cudaFree(d_mappingindices);
   cudaFree(d_Y);
}





