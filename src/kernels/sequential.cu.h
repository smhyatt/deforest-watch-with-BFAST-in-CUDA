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


void ker1seq(uint N, int Kp, int f, int* mappingindices, float* X){
    for (uint i = 0; i < Kp; i++){
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


void ker2naiveseq(uint n, uint N, uint m, float* X, float* XT, float* Y, float* Xsqr, uint K){
    for (uint i = 0; i < m; i++) {              // i  = blockIdx.x
        for (int j1 = 0; j1 < K; j1++) {        // j1 = threadIdx.y
            for (int j2 = 0; j2 < K; j2++) {    // j2 = threadIdx.x
                float acc = 0.0;
                
                for (uint q = 0; q < n; q++) {
                    float y = Y[i*N+q];
                    if (y != F32_MIN) {
                        acc += X[j1*N+q] * XT[q*K+j2] * y;
                    }
                }
                Xsqr[i*K*K + j1*K + j2] = acc;
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


void ker2seqtiled(uint n, uint N, uint m, float* X, float* XT, float* sample, float* Xsqr, uint K, const int R) {
    
    float* YT = (float*) calloc(N*m,sizeof(float));
    transposeMatrix(sample, YT, m, N);

    for (int ii = 0; ii < m; ii+=R) {                           // forall, grid.z
        for (int j1 = 0; j1 < K; j1++) {                        // forall, block.y
            for (int j2 = 0; j2 < K; j2++) {                    // forall, block.x

                float acc[R];                                   // size R, registers

                for (int i = 0; i < R; i++) {                   // fully unroll
                    acc[i] = 0.0;
                }
                float a, b, ab; 
                for (int q = 0; q < n; q++) {
                    a = X[j1*N + q];  
                    b = XT[q*K + j2]; 
                    ab = a*b;

                    for (int i1 = 0; i1 < R; i1++) {            // fully unroll
                        if (ii+i1 < m) {
                            if (YT[q*m + ii+i1] != F32_MIN) {
                            // if (sample[(ii+i1)*N + q] != F32_MIN) {
                                acc[i1] += ab;  
                            }
                        }
                    }

                }
                for (int i2 = 0; i2 < R; i2++) {               // fully unroll
                    if (ii+i2 < m) {
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
void ker3seq(uint M, uint K, float* A, float* AI){

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
            }
        }

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

        // collective copy shared-to-global mem:
        for (int k1 = 0; k1 < K; k1++) {
            for (int k2 = 0; k2 < K; k2++) {
                uint XinvIdx  = k1*(K*2) + k2+K;
                uint XlessIdx = i*K*K + k1*K + k2;
                AI[XlessIdx] = Ash[XinvIdx];
            }
        }
        free(Ash);
        free(AshTmp);
    }
}



/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//// KERNEL 4
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////



void ker4seqnaive(uint m, uint n, uint N, float* X, uint K, 
                                float* sample, float* B0){
    float acc = 0.0;
    for (uint pix = 0; pix < m; pix++) {            // blockIdx.x
        for (int i = 0; i < K; i++) {               // i = threadIdx.y
            acc = 0.0;
            for (uint k = 0; k < n; k++) {
                float cur_y = sample[pix*N+k];

                if (cur_y == F32_MIN) { 
                    acc += 0.0;
                } else {
                    acc += X[i*N+k] * cur_y;
                }
            }
            B0[pix*K + i] = acc;
        }
    }
}    


void ker4seqtiled(uint m, uint n, uint N, float* X, uint K, 
                                float* sample, float* B0){
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
                            if (y != F32_MIN) { 
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




/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//// KERNEL 5
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////


void ker5seqnaive(uint m, float* XsqrInv, uint K, float* B0, float* B){
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


void ker5seqtiled(uint m, float* XsqrInv, uint K, float* B0, float* B){
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


void ker6seqnaive(uint m, uint N, float* XT, float* B, uint K, float* yhat) {
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


void ker6seqtiled(uint m, uint N, float* XT, float* B, uint K, float* yhat) {

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


void filterNaNsWKeys(uint N, float* diffVct, uint* valid, 
                        float* y_errors, int* val_indss) {
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


void ker7seq(uint m, uint N, float* yhat, float* y_errors_all, uint* Nss, 
                        float* y_errors, float* sample, int* val_indss) {
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

        filterNaNsWKeys(N, &y_errors_all[pix*N], &Nss[pix], 
                        &y_errors[pix*N], &val_indss[pix*N]);
    }
}


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//// KERNEL 8
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

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


void ker9seq(uint m, uint N, int* hs, float* y_errors, 
                            uint* nss, float* MO_fsts) {
    int hmax = I32_MIN;
    for (int i = 0; i < m; i++) {
        int cur = hs[i];
        if (cur >= hmax) {
            hmax = cur;
        }
    }

    for (uint pix = 0; pix < m; pix++) {
        for (int i = 0; i < hmax; i++) {
            if (i < hs[pix]) {
                uint idx = i + nss[pix] - hs[pix] + 1;
                MO_fsts[pix] += y_errors[pix*N + idx];
            }
        }
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

void ker10seq(float lam, uint m, uint n, uint N, float* bound, uint* Nss,
           uint* nss, float* sigmas, int* hs, int* mappingindices,
           float* MO_fsts, float* y_errors, int* val_indss, float* MOp,
           float* means, int* fstBreakP, float* MOpp){

    uint Nmn = N-n;

    for (uint s   = 0; s < Nmn; s++){
        uint t    = n+s;
        int time  = mappingindices[t];
        float x   = (float)time / (float)mappingindices[N-1];
        float tmp = x>exp(1.0)? log(x) : 1.0;
        bound[s]  = lam * sqrt(tmp);
    }

    for(int pix = 0; pix < m; pix++) {
        float sigma = sigmas[pix];
        int  Ns = Nss[pix];
        int  ns = nss[pix];
        int  h  = hs[pix];
        float mo = 0.0;
        float mean = 0.0;
        int fstBreak = -1;
        int* val_inds = &val_indss[pix*N];
        for(int i = 0; i < Ns-ns; i++) {
            float tmp;
            if(i==0) {
                tmp = MO_fsts[pix];
            } else {
                tmp = -y_errors[pix*N + ns - h + i] + y_errors[pix*N + ns + i];
            }
            mo += tmp;
            //MO[i] = mo;

            float mop = mo / (sigma * sqrt((float) ns));
            mean += mop;
            if( (fstBreak == -1) && (F32_MIN != mop) && 
                !isnan(mop) && (fabsf(mop) > bound[i]) 
            ) {
                fstBreak = i;
                int adj_break = val_inds[fstBreak+ns] - n;
                fstBreak = ((adj_break-1) / 2) * 2 + 1;
            }
        }
        int fstBreakPP = (ns <=5 || Ns-ns <= 5) ? -2 : fstBreak;
        fstBreakP[pix] = fstBreakPP;
        means[pix]     = mean;
    }

    
#if 0
    float* MO        = (float*) calloc(Nmn*m,sizeof(float));
    int* isBreak     = (int  *) calloc(m,sizeof(int));
    int* fstBreak    = (int  *) calloc(m,sizeof(int));
    int* adjBreak    = (int  *) calloc(m,sizeof(int));
    int* val_indssP  = (int  *) calloc(m*Nmn,sizeof(int));

    for (uint pix = 0; pix < m; pix++){
        float acc = 0.0;
        for (uint i = 0; i < Nmn; i++){
            if(i >= Nss[pix]-nss[pix]){
                MO[pix*Nmn + i] = acc;
            } else if(i==0) {
                acc += MO_fsts[pix];
                MO[pix*Nmn + i] = acc;
            } else {
                acc += -y_errors[pix*N + nss[pix] - hs[pix] + i] + y_errors[pix*N + nss[pix] + i];
                MO[pix*Nmn + i] = acc;
            }
        }

        for (uint i = 0; i < Nmn; i++){
            float mo = MO[pix*Nmn + i];
            MOp[pix*Nmn + i] = mo / (sigmas[pix] * (sqrt( (float) nss[pix] )));
        }

        for (uint i = 0; i < Nmn; i++){
            float mop = MOp[pix*Nmn + i];

            if(i < (Nss[pix]-nss[pix]) && mop != F32_MIN){
                if (fabsf(mop) > bound[i] == 1) {
                    isBreak[pix]  = 1;
                    fstBreak[pix] = i;
                    break;
                }
            }
        }

        for (uint i = 0; i < Nmn; i++) {
            if (i < (Nss[pix]-nss[pix])) {
                means[pix] += MOp[pix*Nmn + i];
            }
        }

        if (!isBreak[pix]){
            fstBreak[pix] = -1;
        } else {
            adjBreak[pix] = (fstBreak[pix] < Nss[pix]-nss[pix])?
                            (val_indss[pix*N + fstBreak[pix] + nss[pix]]-n)
                          : -1;
            fstBreakP[pix] = ((adjBreak[pix]-1) / 2) * 2 + 1;
        }

        if (nss[pix] <= 5 || Nss[pix]-nss[pix] <= 5) {
            fstBreakP[pix] = -2;
        }

        for (int i = 0; i < Nmn; i++) {
            val_indssP[pix*Nmn + i] = (fstBreakP[pix] < Nss[pix]-nss[pix])?
                                      (val_indss[pix*N + fstBreakP[pix]+nss[pix]]-n)
                                    : -1;
        }

        for (int i = 0; i < Nmn; i++) {
            int currIdx = val_indssP[pix*Nmn + i];
            if (currIdx != -1 ) {
                MOpp[pix*Nmn + currIdx] = MOp[pix*Nmn + i];
            }
        }
    }

    free(MO);
    free(isBreak);
    free(fstBreak);
    free(adjBreak);
    free(val_indssP);
#endif
}












