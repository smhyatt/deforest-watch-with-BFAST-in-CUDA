#ifndef KERNELS
#define KERNELS

#include "pbbKernels.cu.h"

#define PI 3.14159265
#define F32_MIN -FLT_MAX
#define I32_MIN -2147483648

typedef unsigned int uint;


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//// KERNEL 1
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

__global__ void ker1(uint N, int K, int freq, int* mappingindices, float* X, float* XT){

    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    int cols = gid % N;
    int rows = gid / N;

    if (gid < N*K) {
        int mi_val = mappingindices[cols];
        float res;

        if(rows == 0){
            res = 1.0;
        } else if(rows == 1){
            res = mi_val;
        } else {
            float angle = 2 * PI * ((float)(rows / 2)) * (float)mi_val / freq;
            if(rows % 2 == 0) {
                res = sin(angle);
            } else {
                res = cos(angle);
            }
        }
        X[gid] = res;
        XT[cols*K + rows]  = X[gid];
    }
}



/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//// KERNEL 2
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////


#if 0
void mkXsqrG(uint n, uint N, uint m, float* X, float* XT, float* sample, float* Xsqr, uint K){
    for (uint pix = 0; pix < m; pix++) {    // pix = blockIdx.x
        for (int i = 0; i < K; i++) {       // i = threadIdx.y
            for (int j = 0; j < K; j++) {   // j = threadIdx.x
                float acc = 0.0;
                for (uint k = 0; k < n; k++) {
                    int mask = isNotNan(sample[pix*N+k]);
                    acc += X[i*N+k] * XT[k*K+j] * mask;
                }
                Xsqr[pix*K*K + i*K + j] = acc;
            }
        }
    }
}
#endif


#if 0
__global__ void ker2(uint n, uint N, uint m, float* X, float* XT, float* sample,
                     float* Xsqr, uint K) {

    int pix = blockIdx.x;
    int i = threadIdx.y;
    int j = threadIdx.x;
    float accum = 0.0f;

    for(int k = 0; k < n; k++) {
        if (sample[pix*N+k] != F32_MIN) {
            accum += X[i*N+k] * XT[k*K+j];
        }
    }

    Xsqr[pix*K*K + i*K + j] = accum;
}
#endif

//
// Cosmin's Matrix Transpose from Weekly 3
//
__global__ void matTransposeTiledKer(float* A, float* B, uint heightA, uint widthA, int T) {
  extern __shared__ char sh_mem1[];
  volatile float* tile = (volatile float*)sh_mem1;
  //__shared__ float tile[T][T+1];

  int x = blockIdx.x * T + threadIdx.x;
  int y = blockIdx.y * T + threadIdx.y;

  if( x < widthA && y < heightA )
      tile[threadIdx.y*(T+1) + threadIdx.x] = A[y*widthA + x];

  __syncthreads();

  x = blockIdx.y * T + threadIdx.x;
  y = blockIdx.x * T + threadIdx.y;

  if( x < heightA && y < widthA )
      B[y*heightA + x] = tile[threadIdx.x*(T+1) + threadIdx.y];
}


__global__ void ker2(uint n, uint N, uint m, float* X, float* XT, float* YT, float* Xsqr, uint K) {

    const int R = 30;

    int ii  = blockIdx.x * R;       // grid.z
    int j1  = threadIdx.y;          // block.y
    int j2  = threadIdx.x;          // block.x
    int lid = threadIdx.y * blockDim.x + threadIdx.x;   // for copying to shared memory

    __shared__ float Yqsh[R];       // shared memory
    float acc[R];                   // registers

    #pragma unroll
    for (int i = 0; i < R; i++) {   // fully unroll
        acc[i] = 0.0;
    }

    float a, b, ab;

    for (int q = 0; q < n; q++) {
        ab = 0.0;
        if (j1 < K && j2 < K) {
            a = X[j1*N + q];       // a = X[j1, q];
            b = XT[q*K + j2];      // b = XT[q, j2];
            ab = a*b;
        }

        // collective copy global-to-shared
        for (int tid = lid; tid < R; tid+=(blockDim.x*blockDim.y)) {
            float tmp = F32_MIN;
            if ((ii+tid) < m) {
                tmp = YT[q*m + (ii+tid)];
                // tmp = YT[(ii+tid)*N + q];
            }
            Yqsh[tid] = tmp;
        }

        __syncthreads();     // block-level synch

        #pragma unroll
        for (int i1 = 0; i1 < R; i1++) { // fully unroll
            if (Yqsh[i1] != F32_MIN) {
                acc[i1] += ab;       // acc[i1] += ab * (1.0-isnan(yqsh[i1]));
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i2 = 0; i2 < R; i2++) { // fully unroll
        if (ii+i2 < m && j1 < K && j2 < K) {
            Xsqr[(ii+i2)*(K*K) + j1*K + j2] = acc[i2];
        }
    }
}


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//// KERNEL 3
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////


__global__ void ker3(uint M, uint K, float* A, float* AI){
    int i = blockIdx.x;
    int k1 = threadIdx.y;
    int k2 = threadIdx.x;

    extern __shared__ float shared[]; // 2*K*K
    float* Ash = &shared[0];
    float* AshTmp = &shared[2*K*K];

    if (k2 < K) {
        // copy the data from the device memory to the first half of the sh_mem
        Ash[k1*2*K + k2] = A[i*K*K + k1*K + k2];
    } else {
        // writes the identity matrix to the second half
        Ash[k1*2*K + k2] = (float) (k1+K == k2);
    }

    // #pragma unroll
    for (uint q = 0; q < K; q++){               // sequential
        float vq = Ash[q];
        // for k1 for k2
        float tmp = 0.0;
        if (vq == 0.0) {
            tmp = Ash[k1*2*K + k2];
        } else {
            float x = Ash[k2] / vq;
            if (k1 == (K-1)){
                tmp = x;
            } else {
                tmp = Ash[(k1+1)*2*K + k2] - Ash[(k1+1)*2*K + q] * x;
            }
        }

        // barrier for block-level sync
        // __syncthreads();
        AshTmp[k1*2*K + k2] = tmp;

        // barrier for block-level sync
        __syncthreads();

        // swap pointers
        float* tmp2 = AshTmp;
        AshTmp = Ash;
        Ash    = tmp2;
    }

    if (K <= k2) {
        AI[i*K*K + k1*K + k2 - K] = Ash[k1*2*K + k2];
    }
}


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//// KERNEL 4
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

#if 0
void mkB0G(uint m, uint n, uint N, float* X, uint K, float* sample, float* B0){
    float acc = 0.0;
    for (uint pix = 0; pix < m; pix++) {            // blockIdx.x
        for (int i = 0; i < K; i++) {               // i = threadIdx.y
            acc = 0.0;
            for (uint k = 0; k < n; k++) {
                float cur_y = sample[pix*N+k];

                if (cur_y == F32_MIN) {             // we only accumulate if y is not nan.
                    acc += 0.0;
                } else {
                    acc += X[i*N+k] * cur_y;
                }
            }
            B0[pix*K + i] = acc;
        }
    }
}
#endif

// __global__ void ker4(uint m, uint n, uint N, float* X, uint K, float* sample, float* B0){

//     int pix = blockIdx.x;
//     int i = threadIdx.y;
//     float accum = 0.0f;

//     for(int k = 0; k < n; k++) {
//         // setting valid bit of valid pixel data
//         float cur_y = sample[pix*N+k];

//         if (cur_y == F32_MIN) {
//             accum += 0.0;
//         } else {
//             accum += X[i*N+k] * cur_y;
//         }
//     }

//     // adding results to beta0
//     B0[pix*K + i] = accum;
// }


// Block-Tiling version without copying from global to shared memory.
__global__ void ker4simple(uint m, uint n, uint N, float* X, uint K, float* Y, float* B0){

    uint pix_out = blockIdx.x * K;
    uint pix_in  = threadIdx.x;
    uint pix     = pix_out + pix_in;
    uint i       = threadIdx.y;         // blocksize K*K
    float acc    = 0.0f;

    for (uint kk = 0; kk < n; kk+=K) {

        // copy to Xsh and to Ysh

        for(int k = 0; k < K; k++) {
            float y = Y[pix*N + (kk+k)];

            if( kk+k < n && pix < m) {
                if (y != F32_MIN) {
                    acc += X[i*N + (kk+k)] * y;
                }
            }
        }
    }

    // adding results to beta0
    if(pix < m && i < K) {
        B0[pix*K + i] = acc;
    }
}

__global__ void ker4(uint m, uint n, uint N, float* X, uint K, float* Y, float* B0){

    uint pix_out = blockIdx.x * K;
    uint pix_in  = threadIdx.y;
    uint pix     = pix_out + pix_in;
    uint i       = threadIdx.x;
    float acc    = 0.0f;
    const uint T = 8;

    __shared__ float Ysh[T][T];  // size K*K
    __shared__ float Xsh[T][T];  // size K*K

    for (uint kk = 0; kk < n; kk+=K) {
        // collective copy global-to-shared of Xsh and to Ysh
        float tmpY = F32_MIN;
        float tmpX = 0.0;

        if (pix < m && kk+threadIdx.x < n) {
            tmpY = Y[pix*N + (kk+threadIdx.x)];
        }

        if (kk+threadIdx.x < n && threadIdx.y < K) {
            tmpX = X[threadIdx.y*N + (kk+threadIdx.x)];
        }

        Ysh[threadIdx.y][threadIdx.x] = tmpY;
        Xsh[threadIdx.y][threadIdx.x] = tmpX;

        __syncthreads();     // block-level synch

        for(int k = 0; k < K; k++) {
            float y = Ysh[pix_in][k]; //pix_in is the local and k was originally (kk+k), but we look locally

            if (y != F32_MIN) {
                acc += Xsh[i][k] * y; // X was indexed on i, and k because it is local
            }
        }
        __syncthreads();
    }

    // adding results to beta0
    if(pix < m && i < K) {
        B0[pix*K + i] = acc;
    }
}




/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//// KERNEL 5
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

#if 0
void ker5seq(uint m, float* XsqrInv, uint K, float* B0, float* B){
    for (uint pix_out = 0; pix_out < m; pix_out+=K) {
        for (uint pix_in = 0; pix_in < K; pix_in++) {
            uint pix = pix_in + pix_out;

            for (int i = 0; i < K; i++) {
                float acc = 0.0;

                for (uint j = 0; j < K; j++) {
                    acc += XsqrInv[pix*(K*K) + i*K + j] * B0[pix*K + j];
                }
                B[pix*K + i] = acc;
            }
        }
    }
}
#endif

__global__ void ker5(uint m, float* Xinv, uint K, float* B0, float* B){
    uint pix_out = blockIdx.x * K;      // pix tiled with K
    uint pix_in  = threadIdx.y;         // local block index
    uint pix     = pix_in + pix_out;
    uint i       = threadIdx.x;
    float acc    = 0.0f;

    for(int k = 0; k < K; k++) {

        if (pix < m && i < K) {
            float beta0 = B0[pix*K + k];
            acc += Xinv[pix*(K*K) + i*K + k] * beta0;
        }
    }
    if (pix < m && i < K) {
        B[pix*K + i] = acc;
    }
}


__global__ void ker5OP(uint m, float* Xinv, uint K, float* B0, float* B){
    uint pix_out = blockIdx.x * K;      // pix tiled with K
    uint pix_in  = threadIdx.y;         // local block index
    uint pix     = pix_in + pix_out;
    uint i       = threadIdx.x;
    float acc    = 0.0f;
    const uint T = 8;

    __shared__ float B0sh[T][T];
    __shared__ float Xinvsh[T][T][T];

    // collective copy global-to-shared of B0
    float tmpB0 = 0.0;

    if (pix < m && threadIdx.x < K) {
        tmpB0 = B0[pix*K + threadIdx.x];
    }

    B0sh[threadIdx.y][threadIdx.x] = tmpB0;


    for (int q = 0; q < K; q++) {
        float tmpXi = 0.0;
        if (pix_out+q < m) {
            // copying in coalesced form, because threadIdx.x is inner.
            tmpXi = Xinv[(pix_out+q)*(K*K) + threadIdx.y*K + threadIdx.x];
        }

        Xinvsh[q][threadIdx.y][threadIdx.x] = tmpXi;
    }

    __syncthreads();     // block-level synch

    for(int k = 0; k < K; k++) {

        if (pix < m && i < K) {
            float beta0 = B0sh[pix_in][k];
            float xinv  = Xinvsh[pix_in][i][k];
            acc += xinv * beta0;
        }
    }

    if (pix < m && i < K) {
        B[pix*K + i] = acc;
    }
}



/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//// KERNEL 6
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////


__global__ void ker6simple(uint m, uint N, float* XT, float* B, uint K, float* yhat){

    uint pix_out = blockIdx.y * K;
    uint pix_in  = threadIdx.y;
    uint ii      = blockIdx.x * K;
    uint i       = threadIdx.x;
    uint pix     = pix_out + pix_in;
    float acc    = 0.0f;

    for(int k = 0; k < K; k++) {

        if(pix < m && i+ii < N) {
            float beta = B[pix*K + k];
            acc += XT[(i+ii)*K + k] * beta;
        }
    }

    if(pix < m && ii+i < N) {
        yhat[pix*N + (ii+i)] = acc;
    }
}



__global__ void ker6(uint m, uint N, float* XT, float* B, uint K, float* yhat){

    uint pix_out = blockIdx.y * K;
    uint pix_in  = threadIdx.y;
    uint ii      = blockIdx.x * K;
    uint i       = threadIdx.x;
    uint pix     = pix_out + pix_in;
    float acc    = 0.0f;
    const uint T = 8;

    __shared__ float Bsh [T][T];  // size K*K
    __shared__ float XTsh[T][T];  // size K*K

    // collective copy global-to-shared of Xsh and to Ysh
    float tmpB  = 0.0;
    float tmpXT = 0.0;

    if (pix < m && threadIdx.x < K) {
        tmpB = B[pix*K + threadIdx.x];
    }

    if (ii+threadIdx.y < N && threadIdx.y < K) {
        tmpXT = XT[(ii+threadIdx.y)*K + threadIdx.x];
    }

    Bsh [threadIdx.y][threadIdx.x] = tmpB;
    XTsh[threadIdx.y][threadIdx.x] = tmpXT;

    __syncthreads();     // block-level synch

    for(int k = 0; k < K; k++) {

        if(pix < m && i+ii < N) {
            float beta = Bsh[pix_in][k];
            acc += XTsh[i][k] * beta;
        }
    }
    __syncthreads();

    // adding results to yhat
    if(pix < m && ii+i < N) {
        yhat[pix*N + (ii+i)] = acc;
    }
}


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//// KERNEL 7
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

__global__ void ker7(uint m, uint N, float* yhat, float* y_errors_all, uint* Nss, float* y_errors, float* Y, int* val_indss) {
    // size 2*N*sizeof(float)
    extern __shared__ volatile uint shmem[];
    volatile int*   shinds = (volatile int*)shmem;
    volatile float* shvals = (volatile float*)(shinds + N);
    // N*sizeof(int),
    uint pix = blockIdx.x;
    uint i = threadIdx.x;
    // for (uint i = 0; i < N; i++) { // i = threadIdx.x; blockDim.x = N
    float y  = Y[pix*N + i];
    float yh = yhat[pix*N + i];

    float y_errors_i;
    if (y != F32_MIN) {
        y_errors_i = y-yh;
    } else {
        y_errors_i = F32_MIN;
    }

    // map \v -> 1 - ...
    uint tf = 0;
    if (y_errors_i != F32_MIN) { tf = 1; }
    shmem[threadIdx.x] = tf;

    int indT = scanIncBlock<Add<int> >(shinds, threadIdx.x);
    int ind = tf * indT - 1;
    shvals[threadIdx.x] = F32_MIN;

    __syncthreads();

    if(ind > -1) {
        shvals[ind] = y_errors_i;
    }

    shinds[threadIdx.x] = 0;

    __syncthreads();

    if (ind > -1) {
        shinds[ind] = threadIdx.x;
    }

    __syncthreads();

    // for result i
    if(threadIdx.x == N-1) {// meaning it is the last thread in the block, holding the value of the reduce
        Nss[pix] = indT;
    }
    // copy vs_sh and ks_sh to global memory => giving coalesced access
    y_errors [pix*N + threadIdx.x] = shvals[threadIdx.x];
    val_indss[pix*N + threadIdx.x] = shinds[threadIdx.x];

}


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//// KERNEL 8
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
#if 0
void ker8naive(uint m, uint n, uint N, uint K, float hfrac, float* y_errors,
               float* y, uint* nss, int* hs, float* sigmas) {
    for (uint pix = 0; pix < m; pix++) {            // parallel blocks
        for (uint i = 0; i < n; i++) {              // parallel threads
            nss[pix] += (y[pix*N + i] != F32_MIN);  // reduce (p) [] nss
        }

        float acc = 0.0;
        for (uint j = 0; j < n; j++) {              // parallel threads
            if (j < nss[pix]) {
                float y_err = y_errors[pix*N + j];
                acc += y_err * y_err;               // reduce (err^2) [] y_err
            }
        }

        hs[pix] = (int)(((float) nss[pix]) * hfrac);
        sigmas[pix] = sqrt(acc / ((float)(nss[pix] - K)));
    }
}
#endif

__global__ void ker8naive(uint m, uint n, uint N, uint K, float hfrac,
                          float* y_errors, float* y, uint* nss, int* hs,
                          float* sigmas) {
    int pix = blockIdx.x;
    int i = threadIdx.x;

    if(i==0) {
        nss[pix] = 0;
    }
    __syncthreads();

    // for (uint pix = 0; pix < m; pix++) {            // parallel blocks
    //     for (uint i = 0; i < n; i++) {              // parallel threads
    nss[pix] += (uint) (y[pix*N + i] != F32_MIN);      // reduce (p) [] nss
    // }
    __syncthreads();

    // float acc = 0.0;
    // // for (uint i = 0; i < n; i++) {              // parallel threads
    // if (i < nss[pix]) {
    //     float y_err = y_errors[pix*N + i];
    //     acc += y_err * y_err;               // reduce (err^2) [] y_err
    // }
    // // }
    // __syncthreads();
    // if(i == blockDim.x) {
    //     hs[pix] = (int)(((float) nss[pix]) * hfrac);
    //     sigmas[pix] = sqrt(acc / ((float)(nss[pix] - K)));
    // }
    // // }
}
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//// KERNEL 9
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

// `N` is the length of the input array
// `T` is the total number of CUDA threads spawned.
// `d_tmp` is the result array, having number-of-blocks elements.
// `d_in` is the input array of length `N`.

// __global__ void ker9(uint m, uint N, int* hs, float* yerrs, uint* nss, float* MOfsts) {
//     extern __shared__ volatile uint shmem[];

//     uint pix = blockIdx.x;
//     uint i = threadIdx.x;

//     // get hmax
//     // redCommuKernel(hmax_sh, hs_sh, N, T);
//     int h = hs[pix*N + i];

//     if (pix < m && threadIdx.x < N) {
//         shmem[threadIdx.x] = h;
//     }

//     int hmax = scanIncBlock<Max<int> >(shmem, threadIdx.x);

//     // RET - KUN MIDLERTIDIG !!!!!!
//     int hpix = hs[pix];
//     uint nsp = nns[pix];
//     if (i < hmax) {
//         MOfsts[pix] += yerrs[pix*N + i + nsp - hpix + 1];
//     }

// }


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//// KERNEL 10
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////



#endif







