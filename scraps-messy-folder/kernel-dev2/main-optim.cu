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

int gpuAssert(cudaError_t code) {
    if(code != cudaSuccess) {
      printf("GPU Error: %s\n", cudaGetErrorString(code));
      return -1;
    }
    return 0;
}

//
// Cosmin's Matrix Transpose Wrapper from Weekly 3
//
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
   cudaThreadSynchronize();
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
   int  K    = 2*k + 2;
   uint m    = 2;
   uint n    = (uint)atoi(input3);
   uint N    = (uint)atoi(input8);
   uint mIRL = (uint)atoi(input7);
   int trend = atoi(input1);
   float freq  = atof(input4);
   float hfrac = atof(input5);
   float lam   = atof(input6);
   const int R = 30;

   int mappingLen, imageLen, i = 0;

   // getting the lengths of mappingindices and images
   while (getc(fp)   != EOF) { mappingLen++; }
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
   float* h_Y = (float*) calloc(N*m,sizeof(float));

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
      h_Y[i] = atof(pixelsPtr);
      i++;
      pixelsPtr = strtok(NULL, delim);
   }

   // closing file with data
   fclose(fp);

   // opening file for validation of results
   FILE* fpV = fopen("../data/val.data","a+");


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

   // allocate host memory for X
   float* h_X      = (float*) calloc(N*K,sizeof(float));
   float* h_XT     = (float*) calloc(K*N,sizeof(float));
   float* h_Xsqr   = (float*) calloc(K*K*m,sizeof(float));
   float* h_B0     = (float*) calloc(K*m,sizeof(float));
   float* h_B      = (float*) calloc(K*m,sizeof(float));
   float* h_yhat   = (float*) calloc(N*m,sizeof(float));
   float* h_Xinv   = (float*) calloc(K*K*m,sizeof(float));
   float* h_yerall = (float*) calloc(m*N,sizeof(float));
   float* h_yerrs  = (float*) calloc(m*N,sizeof(float));
   uint * h_Nss    = (uint *) calloc(m*N,sizeof(uint));
   uint * h_nss    = (uint *) calloc(m,sizeof(uint));
   int  * h_hs     = (int  *) calloc(m,sizeof(int));
   float* h_sigmas = (float*) calloc(m,sizeof(float));
   int  * h_indss  = (int  *) calloc(m*N,sizeof(int));
   float* h_MOfsts = (float*) calloc(m,sizeof(float));

   // allocate device memory for X, XT and Xsqr
   float *d_X, *d_XT, *d_Xsqr, *d_Xinv, *d_YT, *d_B0, *d_B, *d_yhat;
   float *d_yerall, *d_yerrs, *d_MOfsts, *d_sigmas;
   uint  *d_Nss, *d_nss;
   int   *d_indss, *d_hs;
   cudaMalloc((void**) &d_X, X_size);
   cudaMalloc((void**) &d_XT, X_size);
   cudaMalloc((void**) &d_YT, Y_size);
   cudaMalloc((void**) &d_Xsqr, Xsqr_size);
   cudaMalloc((void**) &d_Xinv, Xsqr_size);
   cudaMalloc((void**) &d_B0, B0_size);
   cudaMalloc((void**) &d_B, B0_size);
   cudaMalloc((void**) &d_yhat, Y_size);
   cudaMalloc((void**) &d_yerall, Y_size);
   cudaMalloc((void**) &d_yerrs, Y_size);
   cudaMalloc((void**) &d_Nss, Nss_size);
   cudaMalloc((void**) &d_nss, nss_size);
   cudaMalloc((void**) &d_hs, hs_size);
   cudaMalloc((void**) &d_sigmas, sigmas_size);
   cudaMalloc((void**) &d_indss, I_size);
   cudaMalloc((void**) &d_MOfsts, MO_size);


   /////////////////////////////////////////////////////////////////////////
   //// KERNEL 1
   /////////////////////////////////////////////////////////////////////////
   {
      dim3 block(1024, 1, 1);
      dim3 grid (1024, 1, 1);

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL);

      // GPU call to kernel 1
      ker1 <<< grid, block >>>(N, K, freq, d_mappingindices, d_X, d_XT);
      cudaDeviceSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

      // check for cuda errors
      gpuAssert( cudaPeekAtLastError() );

      // copy result from device to host
      cudaMemcpy(h_X, d_X, X_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_XT, d_XT, X_size, cudaMemcpyDeviceToHost);

      // add to validation
      printX(fpV, h_X, K, N);

      printf("GPU Optimized Kernel 1 runs in: %lu microsecs\n", elapsed);
      // float microsecPerMatrixMul = elapsed;
      // double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A;
      // double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f));
      // printf( "GPU Optimized Kernel 1 Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y);
   }


   /////////////////////////////////////////////////////////////////////////
   //// KERNEL 2
   /////////////////////////////////////////////////////////////////////////
   {
      dim3 block(K, K, 1);
      dim3 grid ((m+R-1)/R, 1, 1);

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL);

      // transpose Y for kernel 2 optimization
      transposeTiled(d_Y, d_YT, m, N, K);
      // GPU call to kernel 2
      ker2 <<< grid, block >>> (n, N, m, d_X, d_XT, d_YT, d_Xsqr, K);
      cudaDeviceSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

      // check for cuda errors
      gpuAssert( cudaPeekAtLastError() );

      // copy result from device to host
      cudaMemcpy(h_Xsqr, d_Xsqr, Xsqr_size, cudaMemcpyDeviceToHost);

      // validation
      printM(fpV, h_Xsqr, m, K);


      printf("GPU Optimized Kernel 2 runs in: %lu microsecs\n", elapsed);
      // float microsecPerMatrixMul = elapsed;
      // double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A;
      // double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f));
      // printf( "GPU Optimized Kernel 2 Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y);
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

      // GPU call to kernel 3
      ker3<<< grid, block, 4*K*K*sizeof(float) >>>(m, K, d_Xsqr, d_Xinv);
      cudaDeviceSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

      // check for cuda errors
      gpuAssert( cudaPeekAtLastError() );

      // copy result from device to host
      cudaMemcpy(h_Xinv, d_Xinv, Xsqr_size, cudaMemcpyDeviceToHost);

      // validation
      printM(fpV, h_Xinv, m, K);


      printf("GPU Optimized Kernel 3 runs in: %lu microsecs\n", elapsed);
      // float microsecPerMatrixMul = elapsed;
      // double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A;
      // double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f));
      // printf( "GPU Optimized Kernel 3 Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y);
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

      // GPU call to kernel 4
      ker4 <<< grid, block >>> (m, n, N, d_X, K, d_Y, d_B0);
      cudaDeviceSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

      // check for cuda errors
      gpuAssert( cudaPeekAtLastError() );

      // copy result from device to host
      cudaMemcpy(h_B0, d_B0, B0_size, cudaMemcpyDeviceToHost);

      // add to validation
      printVf(fpV, h_B0, m, K);

      printf("GPU Optimized Kernel 4 runs in: %lu microsecs\n", elapsed);
      // float microsecPerMatrixMul = elapsed;
      // double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A;
      // double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f));
      // printf( "GPU Optimized Kernel 4 Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y);
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

      // GPU call to kernel 5
      // ker5 <<< grid, block >>> (m, d_Xinv, K, d_B0, d_B);
      ker5OP <<< grid, block >>> (m, d_Xinv, K, d_B0, d_B);
      cudaDeviceSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

      // check for cuda errors
      gpuAssert( cudaPeekAtLastError() );

      // copy result from device to host
      cudaMemcpy(h_B, d_B, B0_size, cudaMemcpyDeviceToHost);

      // validation
      printVf(fpV, h_B, m, K);

      printf("GPU Optimized Kernel 5 runs in: %lu microsecs\n", elapsed);
      // float microsecPerMatrixMul = elapsed;
      // double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A;
      // double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f));
      // printf( "GPU Optimized Kernel 5 Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y);
   }

   /////////////////////////////////////////////////////////////////////////
   //// KERNEL 6
   /////////////////////////////////////////////////////////////////////////
   {
      int  dimx = ceil( ((float) N)/ K);
      int  dimy = ceil( ((float) m)/ K);
      dim3 block(K, K, 1);
      dim3 grid (dimx, dimy, 1);

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL);

      // GPU call to kernel 6
      ker6 <<< grid, block >>> (m, N, d_XT, d_B, K, d_yhat);
      cudaDeviceSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

      // check for cuda errors
      gpuAssert( cudaPeekAtLastError() );

      // copy result from device to host
      cudaMemcpy(h_yhat, d_yhat, Y_size, cudaMemcpyDeviceToHost);

      // validation
      printVf(fpV, h_yhat, m, N);

      printf("GPU Optimized Kernel 6 runs in: %lu microsecs\n", elapsed);
      // float microsecPerMatrixMul = elapsed;
      // double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A;
      // double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f));
      // printf( "GPU Optimized Kernel 6 Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y);
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

      // GPU call to kernel 7
      ker7 <<< grid, block, 2*N*sizeof(float) >>> (m, N, d_yhat, d_yerall, d_Nss, d_yerrs, d_Y, d_indss);
      cudaDeviceSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

      // check for cuda errors
      gpuAssert( cudaPeekAtLastError() );

      // copy result from device to host
      cudaMemcpy(h_Nss, d_Nss, Nss_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_yerrs, d_yerrs, Y_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_indss, d_indss, I_size, cudaMemcpyDeviceToHost);

      // validation
      printE(fpV, h_Nss, m);
      printVfnan(fpV, h_yerrs, m, N);
      printVi(fpV, h_indss, m, N);

      printf("GPU Optimized Kernel 7 runs in: %lu microsecs\n", elapsed);
      // float microsecPerMatrixMul = elapsed;
      // double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A;
      // double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f));
      // printf( "GPU Optimized Kernel 7 Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y);
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

       // GPU call to kernel 8
    //    printf("ker 8 hfrac: %f \n", hfrac);
       ker8optim<<< grid, block, n*sizeof(uint) + n*sizeof(float) >>>(m, n, N, K, hfrac,
                                   d_yerrs, d_Y,
                                   d_nss, d_hs, d_sigmas);
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

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
        float microsecPerMatrixMul = elapsed;
        double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A;
        double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f));
        printf( "GPU Optimized Kernel 8 Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y);
    }

#if 0

    /////////////////////////////////////////////////////////////////////////
    //// KERNEL 9
    /////////////////////////////////////////////////////////////////////////
   {
      dim3 block(N, 1, 1);
      dim3 grid (m, 1, 1);

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL);

      // GPU call to kernel 9
      ker9 <<< grid, block >>> (m, N, d_hs, d_yerrs, d_nss, d_MOfsts);
      cudaDeviceSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

      // check for cuda errors
      gpuAssert( cudaPeekAtLastError() );

      // copy result from device to host
      cudaMemcpy(h_MOfsts, d_MOfsts, _size, cudaMemcpyDeviceToHost);

      // validation
      printEf(fpV, h_MOfsts, m);

      printf("GPU Optimized Kernel 9 runs in: %lu microsecs\n", elapsed);
      // float microsecPerMatrixMul = elapsed;
      // double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A;
      // double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f));
      // printf( "GPU Optimized Kernel 9 Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y);
   }


   /////////////////////////////////////////////////////////////////////////
   //// KERNEL 10
   /////////////////////////////////////////////////////////////////////////
   {
      int  dimx = ceil( ((float) WIDTH_B)/TILE_HEIGHT );
      int  dimy = ceil( ((float)HEIGHT_A)/TILE_WIDTH );
      dim3 block(TILE_WIDTH, TILE_HEIGHT, 1);
      dim3 grid (dimx, dimy, 1);

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL);

      // GPU call to kernel 10
    //   ker10 <<< grid, block >>> ();
      // cudaDeviceSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

      // check for cuda errors
      gpuAssert( cudaPeekAtLastError() );

      // copy result from device to host
      // cudaMemcpy(h_X, d_X, X_size, cudaMemcpyDeviceToHost);

      printf("GPU Optimized Kernel 10 runs in: %lu microsecs\n", elapsed);
      float microsecPerMatrixMul = elapsed;
      double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A;
      double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f));
      printf( "GPU Optimized Kernel 10 Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y);

   }
#endif


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
   free(h_indss);
   free(h_hs);
   free(h_MOfsts);
   cudaFree(d_X);
   cudaFree(d_XT);
   cudaFree(d_YT);
   cudaFree(d_Xsqr);
   cudaFree(d_Xinv);
   cudaFree(d_B0);
   cudaFree(d_B);
   cudaFree(d_yhat);
   cudaFree(d_yerall);
   cudaFree(d_yerrs);
   cudaFree(d_Nss);
   cudaFree(d_indss);
   cudaFree(d_hs);
   cudaFree(d_MOfsts);
   cudaFree(d_mappingindices);
   cudaFree(d_Y);

}


