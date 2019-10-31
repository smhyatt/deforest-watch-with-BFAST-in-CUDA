#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

#include "helper.cu.h"

#define PI 3.141592653589793115997963468544185161590576171875
#define F32_MIN -FLT_MAX
#define I32_MIN -2147483648
typedef unsigned int uint;


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//// KERNEL 1
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////


void mkX(uint N, int kp, int f, int* mappingindices, float* X){
    for (uint i = 0; i < kp; i++){
        for (uint j = 0; j < N; j++){
            float ind = mappingindices[j];
            uint index = i*N + j;
            if(i==0){
                X[index] = 1.0;
            } else if(i==1){
                X[index] = ind;
            } else {
                float ip = (float)(i / 2);
                float jp = ind;
                float angle = 2 * PI * ip * jp / f;
                if(i%2 == 0) {
                    X[index] = sin(angle);
                } else {
                    X[index] = cos(angle);
                }
            }
        }
    }
}


void transpose(uint N, int K, float* X, float* XT) {
    for (uint i = 0; i < K; i++){
        for (uint j = 0; j < N; j++){
            uint Xidx  = i*N + j;
            uint XTidx = j*K + i;
            XT[XTidx]  = X[Xidx];
        }
    }
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//// KERNEL 2
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////


float dotProdFilt(uint n, float* Xvct, float* XTvct, float* yvct) {
    float acc = 0.0;
    for (uint i = 0; i < n; i++) {
        if (yvct[i] != F32_MIN) {
            acc += Xvct[i] * XTvct[i];
        }
    }
    return acc;
}


void mmMulFilt(uint n, uint N, float* X, float* XT, float* y, float* Xsqr, uint K){
    float* tspVct = (float*)calloc(n,sizeof(float));
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            uint XIdx = i*N;
            uint resIdx = i*K + j;

            for (uint l = 0; l < n; l++) {
                uint idx  = l*K + j;
                tspVct[l] = XT[idx];
            }

            Xsqr[resIdx] = dotProdFilt(n, &X[XIdx], tspVct, y);
        }
    }

    // free(tspVct);
}

// -- Xsqr,Xsqr−1:[K][K]f32; β0,β:[K]f32
// let Xsqr = mmMulFilt X[:,:n] XT[:n,:] y[:n] -- ker 2
void mkXsqr(uint n, uint N, uint m, float* X, float* XT, float* sample, float* Xsqr, uint K) {
    for (uint pix = 0; pix < m; pix++) {
        mmMulFilt(n, N, X, XT, &sample[pix*N], &Xsqr[pix*K*K], K);
    }

}

int isNotNan(float x){
    if (F32_MIN == x) return 0;
    else return 1;
}

// the squared MM multiplication in one gathered function
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


void transposeMatrix(float* M, float* MT, uint m, uint N) {
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < N; col++) {
            MT[col*m + row] = M[row*N + col];
        }
    }
}


void mkXsqrOptim(uint n, uint N, uint m, float* X, float* XT, float* sample, float* Xsqr, uint K) {

    float* YT = (float*) calloc(N*m,sizeof(float));
    transposeMatrix(sample, YT, m, N);


    const int R = 30;
    for (int ii = 0; ii < m; ii+=R) {                                  // forall, grid.z
        //for (int jj1 = 0; jj1 < K; jj1+=T1) {                          // forall, grid.y
        //    for (int jj2 = 0; jj2 < K; jj2+=T2){                       // forall, grid.x
                for (int j1 = 0; j1 < K; j1++) {        // forall, block.y
                    for (int j2 = 0; j2 < K; j2++) {    // forall, block.x

                        // float yqsh[R];          // size R, shared memory
                        float acc[R]; //  = calloc(R,sizeof(float));          // size R, registers

                        for (int i = 0; i < R; i++) {                   // fully unroll
                            acc[i] = 0.0;
                        }
                        float a, b, ab; //, y;
                        for (int q = 0; q < n; q++) {
                            a = X[j1*N + q];       // a = X[j1, q];
                            b = XT[q*K + j2];      // b = XT[q, j2];
                            ab = a*b;
                            // collective copy global-to-shared
                            // for (int idx = 0; idx < R; idx++) {
                            //     yqsh[idx] = YT[q, ii]; // YT[q,ii:min(ii+R,M)] ?????????????
                            // }
                            // barrier; // block-level synch

                            for (int i1 = 0; i1 < R; i1++) { // fully unroll
                                if (ii+i1 < m) {
                                    if (YT[q*m + ii+i1] != F32_MIN) {
                                    // if (sample[(ii+i1)*N + q] != F32_MIN) {
                                        acc[i1] += ab;          // acc[i1] += ab * (1.0-isnan(yqsh[i1]));
                                    }
                                }
                            }

                        }
                        for (int i2 = 0; i2 < R; i2++) { // fully unroll
                            if (ii+i2 < m) {
                                // Xsqr[pix*K*K + i*K + j] = acc;
                                Xsqr[(ii+i2)*(K*K) + j1*K + j2] = acc[i2];
                            }
                        }
                    }
                }
            }

}


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//// KERNEL 3
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

// M is the number of pixels
// K is the width and height of each of the matrix in A
// A is the input data [M][K][K]
// Ash is the [K][2*K]
// AI is the inverse A [K][K]
void gaussJordanG(uint M, uint K, float* A, float* AI){

    for (uint i = 0; i < M; i++){
        float* Ash    = (float*) calloc(2*K*K,sizeof(float));
        float* AshTmp    = (float*) calloc(2*K*K,sizeof(float));

        // Pad A with identity matrix to the right
        for (uint k1 = 0; k1 < K; k1++){
            for (uint k2 = 0; k2 < 2*K; k2++){
                if (k2<K) {
                    Ash[k1*2*K + k2] = A[i*K*K + k1*K + k2];
                } else {
                    Ash[k1*2*K + k2] = (float) (k2 == (K+k1));
                }
                // barrier
            }
        }

        // for (uint k1 = 0; k1 < K; k1++){
        //     for (uint k2 = 0; k2 < 2*K; k2++){
        //            Ash[k1*K*2 + k2] = (k2<K) ? A[i*K*K + k1*K + k2] : (k2==K+k1) ? 1.0 : 0.0;
        //     }
        // }


        // Gauss-Jordan Elimination the other version:
        for (uint q = 0; q < K; q++){               // sequential
            float vq = Ash[q];
            for (uint k1 = 0; k1 < K; k1++){        // parallel block.y
                for (uint k2 = 0; k2 < 2*K; k2++){  // parallel block.x
                    float tmp = 0.0;
                    if (vq == 0.0) {
                        tmp = Ash[k1*2*K + k2];
                    } else {
                        float x = Ash[k2] / vq;
                        if (k1 == (K-1)){
                            tmp = x;
                        } else {
                            tmp = Ash[(k1+1)*2*K + k2] - Ash[(k1+1)*2*K + q] *x;
                        }
                    }
                    // barrier
                    AshTmp[k1*2*K + k2] = tmp;
                    // barrier
                }
            }
            // switch pointer
            float* tmp2 = AshTmp;
            AshTmp = Ash;
            Ash = tmp2;
        }
        // // after gauss jordan copies id matrix to AI
        // for (int k1 = 0; k1 < K; k1++) {
        //     for (int k2 = 0; k2 < K; k2++) {
        //         uint XinvIdx  = k1*(K*2) + k2;
        //         uint XlessIdx = i*K*K + k1*K + k2;
        //         AI[XlessIdx] = Ash[XinvIdx];
        //     }
        // }

        // collective copy shared-to-global mem:
        for (int k1 = 0; k1 < K; k1++) {
            for (int k2 = 0; k2 < K; k2++) {
                uint XinvIdx  = k1*(K*2) + k2+K;
                uint XlessIdx = i*K*K + k1*K + k2;
                AI[XlessIdx] = Ash[XinvIdx];
            }
        }
        free(Ash);
    }
}



void gaussJordan(float* XsqrInv, uint cols, uint K){
    // Making the upper triangle
    for (uint row = 0; row < K-1; row++){
        for (uint rowWork = row + 1; rowWork < K; rowWork++){
            float xMult = XsqrInv[rowWork*cols+row] / XsqrInv[row*cols+row];
            for (uint col = row; col < cols; col++){
                uint factorIdx   = row     * cols + col;
                uint elemIdx     = rowWork * cols + col;
                XsqrInv[elemIdx] = XsqrInv[elemIdx] - xMult * XsqrInv[factorIdx];
            }
        }
    }

    // scalling
    for (uint i = 0; i < K; i++) {
        float temp = XsqrInv[i*cols+i];
        for (uint j = i; j < cols; j++){
            XsqrInv[i*cols+j] = XsqrInv[i*cols+j] / temp;
        }
    }

    // Making back substitution
    for (uint row = K-1; row >= 1; row--){
        for (int rowWork = row-1; 0 <= rowWork ; rowWork--){
            float x = XsqrInv[rowWork*cols+row] / XsqrInv[row*cols+row];
            for (uint col = row; col < cols; col++){
                XsqrInv[rowWork*cols+col] = XsqrInv[rowWork*cols+col] - XsqrInv[row*cols+col] * x;
            }
        }
    }
}


void doubleDown(uint m, float* XsqrInv, float* XsqrInvLess, uint K) {
    for (int pix = 0; pix < m; pix++) {
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                uint XinvIdx  = pix*K*(2*K) + i*(K*2) + j+K; // XsqrInv er K længere
                uint XlessIdx = pix*K*K + i*K + j;
                XsqrInvLess[XlessIdx] = XsqrInv[XinvIdx];
            }
        }
    }
}


void mkXsqrIdent(float* Xsqr, float* XsqrInv, uint K) {
    uint cols = 2*K;        // 2*8=16
    uint identIdx = K*cols; // 8*16=128

    for (uint i = 0; i < K; i++){
        for (uint j = 0; j < K; j++){
            // 1*8+0
            uint sqrIdx = i*K + j;
            // 1*16
            uint invIdx = i*cols + j;
            XsqrInv[invIdx] = Xsqr[sqrIdx];
        }
    }

    for (uint i = 0; i < K; i++){
        for (uint j = K; j < identIdx; j+=cols+1){
            uint idx = i*identIdx + j;
            XsqrInv[idx] = 1.0;
        }
    }
}


void mkXsqrInv(uint m, float* Xsqr, float* XsqrInv, uint K){
    uint cols = 2*K;        // 2*8=16
    uint identIdx = K*cols; // 8*16=128

    for (int pix = 0; pix < m; pix++) {
        mkXsqrIdent(&Xsqr[pix*(K*K)], &XsqrInv[pix*identIdx], K);
        gaussJordan(&XsqrInv[pix*identIdx], cols, K);
    }

    // for (uint ind = 0; ind < identIdx; ind++){
    //     // (i, j) = (ind / m, ind % m)
    //     uint iIdx = ind / cols;
    //     uint jIdx = ind % cols;
    //     uint invIdx = iIdx*identIdx*sizeof(float) + jIdx*sizeof(float);
    //     uint sqrIdx = iIdx*K*sizeof(float) + jIdx*sizeof(float);
    //     if(jIdx<K){
    //         XsqrInv[invIdx] = Xsqr[sqrIdx];
    //     } else {
    //         if(jIdx==K+iIdx){
    //             XsqrInv[invIdx] = 1.0;
    //         } else {
    //             XsqrInv[invIdx] = 0.0;
    //         }
    //     }
    // }

}


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//// KERNEL 4
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

#if 0
void mvMulFilt(uint n, uint N, float* X, float* y, uint K, float* B0){

    for (int i = 0; i < K; i++) {
        uint XIdx  = i*N;

        B0[i] = dotProdFilt(n, &X[XIdx], y, y);
    }
}
void mkB0(uint m, uint n, uint N, float* X, uint K, float* sample, float* B0){
    for (uint pix = 0; pix < m; pix++) {
        mvMulFilt(n, N, X, &sample[pix*N], K, &B0[pix*K]);
    }
}
#endif

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


void mkBOPN(uint m, uint n, uint N, float* X, uint K, float* sample, float* B0){
    float acc = 0.0;
    for (uint pix_out = 0; pix_out < m; pix_out+=K) {   // blockIdx.x*K
        for (int pix_in = 0; pix_in < K; pix_in++) {    // pix_in = threadIdx.x
            uint pix = pix_out + pix_in;
            for (int i = 0; i < K; i++) {               // i = threadIdx.y
                acc = 0.0;
                for (uint kk = 0; kk < n; kk+=K) {
                    // copy to Xsh and to Ysh
                    for(uint k=0; k < K; k++) {
                        if(kk+k < n && pix < m) {
                            float y = sample[pix*N+(kk+k)];
                            if (y != F32_MIN) {             // we only accumulate if y is not nan.
                                acc += X[i*N+(kk+k)] * y;
                            }
                        }
                    }
                }
                if(pix < m) {
                    B0[pix*K + i] = acc;
                }
            }
        }
    }
}


// void ker4MkB0(uint m, uint n, uint N, float* X, uint K, float* Y, float* B0) {

//     for (int i = 0; i < m; i+=(K*K)) {                  // K*K, it. through pixels ??????????????
//         for (int j = 0; j < K; j+=1) {                  // block


//             for (int kk = 0; kk < n; kk+=K) {
//                 // parallel version does copy to shared memory
//                 // Y[i,k] X[j,k]

//                 for (int k = 0; k < K; k++) {           // fully unroll

//                     float y = Y[(i+k)* N + kk];         // Y[i,k]
//                     if (y != F32_MIN) {                 // we only accumulate if y is not nan.
//                         acc[k] += X[i*N+k] * y;         // Xsh[j,k] * Ysh[?,k] - hvor du laver et tjek om dens validitet inden
//                     }
//                 }
//             }

//             for (int i1 = 0; i1 < K; i1++) {
//                 // B0[pix*K + i] = acc;
//                 B0[(i+j)*N + i1] = acc[i1]; // index ?????????
//                 // B0[(i+i1)*K] = acc[i1]; // index ?????????
//             }

//         }
//     }
// }



/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//// KERNEL 5
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////


// void mvMul(float* M, float* v, uint rows, uint cols, float* res_v) {

//     // for ker6 er rows = N og cols = K
//     for (int i = 0; i < rows; i++) {
//         float acc = 0.0;

//         for (uint elm = 0; elm < cols; elm++) {
//             uint MIdx = i*cols + elm;
//             acc += M[MIdx] * v[elm];
//         }
//         res_v[i] = acc;
//     }
// }

// void ker5seq(uint m, float* XsqrInvLess, uint K, float* B0, float* B){
//     for (uint pix = 0; pix < m; pix++) {
//         mvMul(&XsqrInvLess[pix*(K*K)], &B0[pix*K], K, K, &B[pix*K]);
//     }
// }



void mkB(uint m, float* XsqrInv, uint K, float* B0, float* B){
    for (uint pix = 0; pix < m; pix++) {
        for (int i = 0; i < K; i++) {
            float acc = 0.0;

            for (uint j = 0; j < K; j++) {
                acc += XsqrInv[pix*(K*K) + i*K + j] * B0[pix*K + j];
            }
            B[pix*K + i] = acc;
        }
    }
}


void ker5seq(uint m, float* XsqrInv, uint K, float* B0, float* B){
    for (uint pix_out = 0; pix_out < m; pix_out+=K) {
        for (uint pix_in = 0; pix_in < K; pix_in++) {
            uint pix = pix_in + pix_out;

            for (int i = 0; i < K; i++) {
                float acc = 0.0;

                for (uint j = 0; j < K; j++) {
                    if (pix < m) {
                        acc += XsqrInv[pix*(K*K) + i*K + j] * B0[pix*K + j];
                    }
                }
                if(pix < m) {
                    B[pix*K + i] = acc;
                }
            }
        }
    }
}


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//// KERNEL 6
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

#if 0
void ker6(uint m, uint N, float* XT, float* B, uint K, float* yhat) {
    for (uint pix = 0; pix < m; pix++) {
        mvMul(XT, &B[pix*K], N, K, &yhat[pix*N]);
    }
}
#endif


void ker6seq(uint m, uint N, float* XT, float* B, uint K, float* yhat) {
    for (int pix = 0; pix < m; pix++) {

        for (int i = 0; i < N; i++) {
            float acc = 0.0;

            for (int k = 0; k < K; k++) {
                acc += XT[i*K + k] * B[pix*K + k];
            }
            yhat[pix*N + i] = acc;
        }
    }
}


void ker6seqOP(uint m, uint N, float* XT, float* B, uint K, float* yhat) {

    for (uint pix_out = 0; pix_out < m; pix_out+=K) {

        for (int ii = 0; ii < N; ii+=K) {

            for (uint pix_in = 0; pix_in < K; pix_in++) {
                uint pix = pix_out + pix_in;

                for (int i = 0; i < K; i++) {
                    float acc = 0.0;
                    for (int k = 0; k < K; k++) {

                        if(pix < m && i+ii < N) {
                            acc += XT[(i+ii)*K + k] * B[pix*K + k];
                        }
                    }
                    if(pix < m && ii+i < N) {
                        yhat[pix*N + (ii+i)] = acc;
                    }
                }
            }
        }
    }
}



/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//// KERNEL 7
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////


void filterNaNsWKeys(uint N, float* diffVct, uint* valid, float* y_errors, int* val_indss) {
    uint idx = 0;
    *valid   = 0;

    for (int i = 0; i < N; i++) {
        uint check = (diffVct[i] != F32_MIN);
        *valid    += check;
        int ind    = (check * (*valid) - 1);

        if (ind != -1) {
            y_errors[idx]  = diffVct[i];
            val_indss[idx] = i;
            idx++;
        }
    }

}


void ker7seq(uint m, uint N, float* yhat, float* y_errors_all, uint* Nss, float* y_errors, float* sample, int* val_indss) {
    for (uint pix = 0; pix < m; pix++) {
        for (uint i = 0; i < N; i++) {
            float y  = sample[pix*N+i];
            float yh = yhat[pix*N + i];

            if (y != F32_MIN) {
                y_errors_all[pix*N + i] = y-yh;
            } else {
                y_errors_all[pix*N + i] = F32_MIN;
            }
        }

        filterNaNsWKeys(N, &y_errors_all[pix*N], &Nss[pix], &y_errors[pix*N], &val_indss[pix*N]);

    }
}


// void ker7G(uint m, uint N, float* yhat, float* y_errors_all, uint* Nss, float* y_errors, float* sample, int* val_indss) {
//     for (uint pix = 0; pix < m; pix++) {
//         for (uint i = 0; i < N; i++) {
//             float y  = sample[pix*N+i];
//             float yh = yhat[pix*N + i];

//             if (y != F32_MIN) {
//                 y_errors_all[pix*N + i] = y-yh;
//             } else {
//                 y_errors_all[pix*N + i] = F32_MIN;
//             }
//         }

//         filterNaNsWKeys(N, &y_errors_all[pix*N], &Nss[pix], &y_errors[pix*N], &val_indss[pix*N]);
//         void filterNaNsWKeys(uint N, float* diffVct, uint* valid, float* y_errors, int* val_indss)

//         uint idx = 0;
//         *valid   = 0;

//         for (int i = 0; i < N; i++) {
//             uint check = (diffVct[i] != F32_MIN);
//             *valid    += check;
//             int ind    = (check * (*valid) - 1);

//             if (ind != -1) {
//                 y_errors[idx]  = diffVct[i];
//                 val_indss[idx] = i;
//                 idx++;
//             }
//         }
//     }
// }


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//// KERNEL 8
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

void comp(uint n, float hfrac, float* yh, float* y_errors, uint K, int* hs, uint* nss, float* sigmas) {
    float acc = 0.0;

    for (uint i = 0; i < n; i++) {
        *nss += (yh[i] != F32_MIN);
    }

    for (uint j = 0; j < n; j++) {
        if (j < *nss) {
            float y_err = y_errors[j];
            acc += y_err*y_err;
        }
    }
    *sigmas = sqrt(acc/((float)(*nss-K)));
    *hs = (int)(((float)*nss) * hfrac);
}



void ker8(uint m, uint n, uint N, float hfrac, float* y_errors, uint K, int* hs, uint* nss, float* sigmas, float* sample) {
    for (uint pix = 0; pix < m; pix++) {
        comp(n, hfrac, &sample[pix*N], &y_errors[pix*N], K, &hs[pix], &nss[pix], &sigmas[pix]);
    }
}

void ker8seq(uint m, uint n, uint N, uint K, float hfrac, float* y_errors,
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
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//// KERNEL 9
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

void MO_fsts_comp(int hmax, int* hs, float* y_errors, uint* nss, float* MO_fsts) {

    for (int i = 0; i < hmax; i++) {

        if (i < *hs) {
            uint idx = i + *nss - *hs + 1;
            *MO_fsts += y_errors[idx];
        }
    }
}


void ker9seq(uint m, uint N, int* hs, float* y_errors, uint* nss, float* MO_fsts) {
    int hmax = I32_MIN;
    for (int i = 0; i < m; i++) {
        int cur = hs[i];
        if (cur >= hmax) {
            hmax = cur;
        }
    }

    for (uint pix = 0; pix < m; pix++) {
        MO_fsts_comp(hmax, &hs[pix], &y_errors[pix*N], &nss[pix], &MO_fsts[pix]);
    }
}



/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//// KERNEL 10
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////


float logplus(float x){
    if(x>exp(1.0)){
        return log(x);
    } else {
        return 1.0;
    }
}


void compBound(float lam, uint n, uint N, uint Nmn, int* mappingindices, float* b){
    for (uint i = 0; i < Nmn; i++){
        uint t = n+i;
        int time = mappingindices[t];
        float tmp = logplus((float)time / (float)mappingindices[N-1]);
        b[i] = lam * sqrt(tmp);
    }
}

void MO_comp(uint Nmn, int* h, float* MO_fst, float* y_error, uint* Ns,
             uint* ns, float* MO){
    float acc = 0.0;
    for (uint i = 0; i < Nmn; i++){
        if(i >= *Ns-*ns){
            MO[i] = acc;
        } else if(i==0) {
            acc += *MO_fst;
            MO[i] = acc;
        } else {
            acc += - y_error[*ns - *h + i] + y_error[*ns + i];
            MO[i] = acc;
        }
    }
}


void MO_prime_comp(uint Nmn, float* MO, uint* ns, float sigma, float* MOp){
    for (uint i = 0; i < Nmn; i++){
        float mo = MO[i];
        MOp[i] = mo / (sigma * (sqrt( (float)*ns )));
    }
}


//           let fst_break' = if !is_break then -1
//                              else let adj_break = adjustValInds n ns Ns val_inds fst_break
//                                   in  ((adj_break-1) / 2) * 2 + 1  -- Cosmin's validation hack
//             let fst_break' = if ns <=5 || Ns-ns <= 5 then -2 else fst_break'
void breaks(float* MOp, float* bound, uint Ns, uint ns, uint Nmn, int* isBreak, int* fstBreak){
    for (uint i = 0; i < Nmn; i++){
        float mop = MOp[i];

        if(i < (Ns-ns) && mop != F32_MIN){
            if (fabsf(mop) > bound[i] == 1) {
                *isBreak  = 1;
                *fstBreak = i;
                break;
            }
        }
    }
}


void meanComp(uint Ns, uint ns, uint Nmn, float* MOp, float* mean) {
    for (uint i = 0; i < Nmn; i++) {
        if (i < (Ns-ns)) {
            *mean += MOp[i];
        }
    }
}


// let adjustValInds [N] (n : i32) (ns : i32) (Ns : i32) (val_inds : [N]i32) (ind: i32) : i32 =
//     if ind < Ns - ns then (unsafe val_inds[ind+ns]) - n else -1
int adjustValInds(uint n, uint ns, uint Ns, int* val_inds, int fstBreak) {
    if (fstBreak < Ns-ns) {
        return (val_inds[fstBreak+ns]-n);
    } else {
        return -1;
    }
}



//             let val_inds' = map (adjustValInds n ns Ns val_inds) (iota Nmn)
//             let MO'' = scatter (replicate Nmn f32.nan) val_inds' MO'
//             in (MO'', MO', fst_break', mean)
void fstPComp(uint n, uint ns, uint Ns, int* val_inds, int* isBreak, int* fstBreak, int* adjBreak, int* fstBreakP) {
    if (!isBreak){
        *fstBreakP = -1;
    } else {
        *adjBreak = adjustValInds(n, ns, Ns, val_inds, *fstBreak);
        *fstBreakP = ((*adjBreak-1) / 2) * 2 + 1;
    }

    if (ns <= 5 || Ns-ns <= 5) {
        *fstBreakP = -2;
    }
}


void valIndsPComp(uint n, uint Nmn, uint* ns, uint* Ns, int* fstBreak, int* val_inds, int* val_indsP) {
    for (int i = 0; i < Nmn; i++) {
        val_indsP[i] = adjustValInds(n, *ns, *Ns, val_inds, i);
    }
}


void MOppComp(uint Nmn, float* MOp, int* val_indsP, float* MOpp) {
    for (int i = 0; i < Nmn; i++) {
        int currIdx = val_indsP[i];
        if (currIdx != -1 ) {
            MOpp[ currIdx ] = MOp[i];
        }
    }
}


void ker10seq(float lam, uint m, uint n, uint N, float* bound, uint* Nss,
		   uint* nss, float* sigmas, int* hs, int* mappingindices,
		   float* MO_fsts, float* y_errors, int* val_indss, float* MOp,
           float* means, int* fstBreakP, float* MOpp){

    uint Nmn = N-n;
    compBound(lam, n, N, Nmn, mappingindices, bound);
    float* MO        = (float*) calloc(Nmn*m,sizeof(float));
    int* isBreak     = (int*) calloc(m,sizeof(int));
    int* fstBreak    = (int*) calloc(m,sizeof(int));
    int* adjBreak    = (int*) calloc(m,sizeof(int));
    int* val_indssP  = (int*) calloc(m*Nmn,sizeof(int));

    for (uint pix = 0; pix < m; pix++){
        MO_comp(Nmn, &hs[pix], &MO_fsts[pix], &y_errors[pix*N], &Nss[pix], &nss[pix], &MO[pix*Nmn]);

        MO_prime_comp(Nmn, &MO[pix*Nmn], &nss[pix], sigmas[pix], &MOp[pix*Nmn]);

        breaks(&MOp[pix*Nmn], bound, Nss[pix], nss[pix], Nmn, &isBreak[pix], &fstBreak[pix]);

        meanComp(Nss[pix], nss[pix], Nmn, &MOp[pix*Nmn], &means[pix]);

        fstPComp(n, nss[pix], Nss[pix], &val_indss[pix*N], &isBreak[pix], &fstBreak[pix], &adjBreak[pix], &fstBreakP[pix]);

        valIndsPComp(n, Nmn, &nss[pix], &Nss[pix], &fstBreak[pix], &val_indss[pix*N], &val_indssP[pix*Nmn]);

        MOppComp(Nmn, &MOp[pix*Nmn], &val_indssP[pix*Nmn], &MOpp[pix*Nmn]);
    }

    free(MO);
    free(isBreak);
    free(fstBreak);
    free(adjBreak);
    free(val_indssP);
}




