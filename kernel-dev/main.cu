#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <time.h> 

#include "helper.cu.h"
#include "kernels.cu.h"
#include "sequential.cu.h"

#define WIDTH_A  1024//1024 //1024//2048
#define HEIGHT_A 1024//2048//2048//2048
#define WIDTH_B  4096//2048
#define TILE     16
#define PI 3.14159265
#define F32_MIN -FLT_MAX
#define I32_MIN -2147483648
typedef unsigned int uint;


/////////////////////////////////////////////////////////
// Helpers
/////////////////////////////////////////////////////////

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

/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////
 
int main(int argc, char const *argv[]) {
   if (argc != 2) {
      printf("Please include the name of the dataset.\n");
         return -1;
   }

// *****************************************************************************
// Parsing
// *****************************************************************************
   
   FILE *fp, *fpim;
   
   if (argv[1][0] == 's') {
      fp   = fopen("data/saharaC.in", "r");
      fpim = fopen("data/saharaCimages.in", "r");
   } else {
      fp   = fopen("data/peruC.in", "r");
      fpim = fopen("data/peruCimages.in", "r");
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
   char c;

   // getting the lengths of mappingindices and images
   while ((c = getc(fp)) != EOF) { mappingLen++; }
   while ((c = getc(fpim)) != EOF) { imageLen++; }

   // rewinding the pointer to extract the data
   rewind(fpim);

   // extracting each array
   char mappings[mappingLen], pixels[(imageLen-mappingLen)];
   fscanf(fpim, " %[^\n]  %[^\n] ", mappings, pixels);

   // converting mappingindices from char* to int*
   char delim[] = ",";
   char *mapPtr = strtok(mappings, delim);
 
   // allocating host memory for mappingindices and pixels
   int* h_mappingindices = calloc(N,sizeof(int));
   float* h_sample = calloc(N*m,sizeof(float));

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

   fclose(fp);

    
   // allocate device memory
   uint map_size = N*m*sizeof(float);
   uint sam_size = N*sizeof(float);
   float* d_mappingindices;
   float* d_sample;
   cudaMalloc((void**) &d_mappingindices, map_size);
   cudaMalloc((void**) &d_sample, sam_size);
 
   // copy host memory to device
   cudaMemcpy(d_mappingindices, h_mappingindices, map_size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_sample, h_sample, sam_size, cudaMemcpyHostToDevice);
 
   // allocate host memory for X
   uint X_size    = K*N*sizeof(float);
   float* h_X     = calloc(K*N,sizeof(float));
   float* h_seq_X = calloc(K*N,sizeof(float));
 
   // allocate device memory for X
   float *d_X;
   cudaMalloc((void**) &d_X, X_size);

 
   // compute sequential creation of X and XT
   {
      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL); 
      
      // calling sequential kernel 1 and transpose from the sequential file 
      ker1(N, K, freq, h_mappingindices, h_X);
      // transpose(N, K, h_X, h_XT);
      // matMult<float>(h_A, h_B, seq_C, WIDTH_A, HEIGHT_A, WIDTH_B);

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
      printf("Sequential kernel 1 version runs in: %lu microsecs\n", elapsed);
   }

   
   // execute the block+register tiled kernel
   // ToDo: please fill in the implementation below
   //       (for TILE = 16)
   // {
   //    // 1. you would probably want to compute some valid grid and block here
   //    int  dimy = (HEIGHT_A+TILE-1)/TILE; 
   //    int  dimx = (WIDTH_B+(TILE*TILE)-1)/(TILE*TILE);
   //    dim3 block(TILE, TILE, 1);
   //    dim3 grid (dimx, dimy, 1);

   //    unsigned long int elapsed;
   //    struct timeval t_start, t_end, t_diff;
   //    gettimeofday(&t_start, NULL); 
      
   //    // 2. you would probably want to call here the kernel: 
   //    // matMultRegTiledKer<float,TILE> <<< grid, block >>>(d_A, d_B, d_C, HEIGHT_A, WIDTH_B, WIDTH_A); 
   //    cudaThreadSynchronize();

   //    gettimeofday(&t_end, NULL);
   //    timeval_subtract(&t_diff, &t_end, &t_start);
   //    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 

   //    // copy result from device to host
   //    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
   //    // validate
   //    printf("GPU Block+Register Tiled MMM version ... ");
   //    validate<float>(seq_C, h_C, size_C);

   //    printf("GPU Block+Register Tiled MMM version runs in: %lu microsecs\n", elapsed);
   //    float microsecPerMatrixMul = elapsed; 
   //    double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A; 
   //    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f)); 
   //    printf( "GPU Block+Register Tiled MMM Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y); 
   // }


   // 7. clean up memory
   free(h_mappingindices);
   free(h_sample);
   free(h_seq_X);
   free(h_X);
   cudaFree(d_X);
   cudaFree(d_mappingindices);
   cudaFree(d_sample);

}



