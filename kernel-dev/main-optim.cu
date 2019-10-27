#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <time.h>

#include "helper.cu.h"
#include "kernels-optim.cu.h"
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


   // allocate device memory
   uint map_size = N*sizeof(int);
   uint sam_size = N*m*sizeof(float);
   int* d_mappingindices;
   float* d_sample;
   cudaMalloc((void**) &d_mappingindices, map_size);
   cudaMalloc((void**) &d_sample, sam_size);

   // copy host memory to device
   cudaMemcpy(d_mappingindices, h_mappingindices, map_size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_sample, h_sample, sam_size, cudaMemcpyHostToDevice);

   uint X_size     = K*N*sizeof(float);
   uint Xsqr_size  = K*K*m*sizeof(float);
   uint Xinv_size  = K*K*m*sizeof(float);
   uint B0_size    = K*m*sizeof(float);

   // allocate host memory for X
   float* h_X      = (float*) calloc(N*K,sizeof(float));
   float* h_XT     = (float*) calloc(K*N,sizeof(float));
   float* h_Xsqr   = (float*) calloc(K*K*m,sizeof(float));
   float* h_Xinv   = (float*) calloc(K*K*m,sizeof(float));
   float* h_B0     = (float*) calloc(K*m,sizeof(float));

   // allocate device memory for X, XT and Xsqr
   float *d_X, *d_XT, *d_Xsqr, *d_Xinv, *d_B0;
   cudaMalloc((void**) &d_X, X_size);
   cudaMalloc((void**) &d_XT, X_size);
   cudaMalloc((void**) &d_Xsqr, Xsqr_size);
   cudaMalloc((void**) &d_Xinv, Xinv_size);
   cudaMalloc((void**) &d_B0, B0_size);


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
      float microsecPerMatrixMul = elapsed;
      double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A;
      // double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f));
      // printf( "GPU Optimized Kernel 1 Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y);
   }


   /////////////////////////////////////////////////////////////////////////
   //// KERNEL 2
   /////////////////////////////////////////////////////////////////////////
   {
      dim3 block(K, K, 1);
      dim3 grid (m, 1, 1);

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL);

      // GPU call to kernel 2
      ker2 <<< grid, block >>> (n, N, m, d_X, d_XT, d_sample, d_Xsqr, K);
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
      float microsecPerMatrixMul = elapsed;
      double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A;
      // double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f));
      // printf( "GPU Optimized Kernel 2 Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y);
   }

   /////////////////////////////////////////////////////////////////////////
   //// KERNEL 3
   /////////////////////////////////////////////////////////////////////////
   {
    dim3 block(K, K, 1);
    dim3 grid (n, 1, 1);

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    // GPU call to kernel 3
    ker3<<< grid, block >>>(m, K, d_Xsqr, d_Xinv);
    // cudaDeviceSynchronize();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

    // check for cuda errors
    gpuAssert( cudaPeekAtLastError() );

    // copy result from device to host
    cudaMemcpy(h_Xinv, d_Xinv, Xinv_size, cudaMemcpyDeviceToHost);
    printM(fpV, h_Xinv, m, K);
    // printM(fpV, h_Xsqr, m, K);

    printf("GPU Optimized Kernel 3 runs in: %lu microsecs\n", elapsed);
    float microsecPerMatrixMul = elapsed;
    double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f));
    // printf( "GPU Optimized Kernel 3 Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y);
   }


   /////////////////////////////////////////////////////////////////////////
   //// KERNEL 4
   /////////////////////////////////////////////////////////////////////////
   {
      dim3 block(K, K, 1);
      dim3 grid (m, 1, 1);

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL);

      // GPU call to kernel 4
      ker4 <<< grid, block >>> (m, n, N, d_X, K, d_sample, d_B0);
      cudaDeviceSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

      // check for cuda errors
      gpuAssert( cudaPeekAtLastError() );

      // copy result from device to host
      cudaMemcpy(h_B0, d_B0, B0_size, cudaMemcpyDeviceToHost);

      // add to validation
    //   printVf(fpV, h_B0, m, K);

      printf("GPU Optimized Kernel 4 runs in: %lu microsecs\n", elapsed);
      float microsecPerMatrixMul = elapsed;
      double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A;
      double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f));
      // printf( "GPU Optimized Kernel 4 Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y);
   }

   /////////////////////////////////////////////////////////////////////////
   //// KERNEL 5
   /////////////////////////////////////////////////////////////////////////
   {
      int  dimx = ceil( ((float) WIDTH_B)/TILE_HEIGHT );
      int  dimy = ceil( ((float)HEIGHT_A)/TILE_WIDTH );
      dim3 block(TILE_WIDTH, TILE_HEIGHT, 1);
      dim3 grid (dimx, dimy, 1);

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL);

      // GPU call to kernel 5
      // ker5 <<< grid, block >>> ();
      // cudaDeviceSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

      // check for cuda errors
      gpuAssert( cudaPeekAtLastError() );

      // copy result from device to host
      // cudaMemcpy(h_X, d_X, X_size, cudaMemcpyDeviceToHost);


      printf("GPU Optimized Kernel 5 runs in: %lu microsecs\n", elapsed);
      float microsecPerMatrixMul = elapsed;
      double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A;
      double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f));
      printf( "GPU Optimized Kernel 5 Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y);
   }


   /////////////////////////////////////////////////////////////////////////
   //// KERNEL 6
   /////////////////////////////////////////////////////////////////////////
   {
      int  dimx = ceil( ((float) WIDTH_B)/TILE_HEIGHT );
      int  dimy = ceil( ((float)HEIGHT_A)/TILE_WIDTH );
      dim3 block(TILE_WIDTH, TILE_HEIGHT, 1);
      dim3 grid (dimx, dimy, 1);

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL);

      // GPU call to kernel 6
      // ker6 <<< grid, block >>> ();
      // cudaDeviceSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

      // check for cuda errors
      gpuAssert( cudaPeekAtLastError() );

      // copy result from device to host
      // cudaMemcpy(h_X, d_X, X_size, cudaMemcpyDeviceToHost);

      printf("GPU Optimized Kernel 6 runs in: %lu microsecs\n", elapsed);
      float microsecPerMatrixMul = elapsed;
      double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A;
      double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f));
      printf( "GPU Optimized Kernel 6 Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y);
   }


   /////////////////////////////////////////////////////////////////////////
   //// KERNEL 7
   /////////////////////////////////////////////////////////////////////////
   {
      int  dimx = ceil( ((float) WIDTH_B)/TILE_HEIGHT );
      int  dimy = ceil( ((float)HEIGHT_A)/TILE_WIDTH );
      dim3 block(TILE_WIDTH, TILE_HEIGHT, 1);
      dim3 grid (dimx, dimy, 1);

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL);

      // GPU call to kernel 7
      // ker7 <<< grid, block >>> ();
      // cudaDeviceSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

      // check for cuda errors
      gpuAssert( cudaPeekAtLastError() );

      // copy result from device to host
      // cudaMemcpy(h_X, d_X, X_size, cudaMemcpyDeviceToHost);

      printf("GPU Optimized Kernel 7 runs in: %lu microsecs\n", elapsed);
      float microsecPerMatrixMul = elapsed;
      double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A;
      double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f));
      printf( "GPU Optimized Kernel 7 Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y);
   }


   /////////////////////////////////////////////////////////////////////////
   //// KERNEL 8
   /////////////////////////////////////////////////////////////////////////
   {
      int  dimx = ceil( ((float) WIDTH_B)/TILE_HEIGHT );
      int  dimy = ceil( ((float)HEIGHT_A)/TILE_WIDTH );
      dim3 block(TILE_WIDTH, TILE_HEIGHT, 1);
      dim3 grid (dimx, dimy, 1);

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL);

      // GPU call to kernel 8
      // ker8 <<< grid, block >>> ();
      // cudaDeviceSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

      // check for cuda errors
      gpuAssert( cudaPeekAtLastError() );

      // copy result from device to host
      // cudaMemcpy(h_X, d_X, X_size, cudaMemcpyDeviceToHost);

      printf("GPU Optimized Kernel 8 runs in: %lu microsecs\n", elapsed);
      float microsecPerMatrixMul = elapsed;
      double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A;
      double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f));
      printf( "GPU Optimized Kernel 8 Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y);
   }


   /////////////////////////////////////////////////////////////////////////
   //// KERNEL 9
   /////////////////////////////////////////////////////////////////////////
   {
      int  dimx = ceil( ((float) WIDTH_B)/TILE_HEIGHT );
      int  dimy = ceil( ((float)HEIGHT_A)/TILE_WIDTH );
      dim3 block(TILE_WIDTH, TILE_HEIGHT, 1);
      dim3 grid (dimx, dimy, 1);

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL);

      // GPU call to kernel 9
      // ker9 <<< grid, block >>> ();
      // cudaDeviceSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

      // check for cuda errors
      gpuAssert( cudaPeekAtLastError() );

      // copy result from device to host
      // cudaMemcpy(h_X, d_X, X_size, cudaMemcpyDeviceToHost);

      printf("GPU Optimized Kernel 9 runs in: %lu microsecs\n", elapsed);
      float microsecPerMatrixMul = elapsed;
      double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A;
      double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f));
      printf( "GPU Optimized Kernel 9 Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y);
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
      // ker10 <<< grid, block >>> ();
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


   /////////////////////////////////////////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////
   //// VALIDATION
   /////////////////////////////////////////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////

   fclose(fpV);

   // 7. clean up memory
   free(h_mappingindices);
   free(h_sample);
   free(h_X);
   free(h_XT);
   free(h_Xsqr);
   cudaFree(d_X);
   cudaFree(d_XT);
   cudaFree(d_Xsqr);
   cudaFree(d_mappingindices);
   cudaFree(d_sample);

}


