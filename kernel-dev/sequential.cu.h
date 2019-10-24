#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

#include "helper.cu.h"

#define PI 3.14159265
#define F32_MIN -FLT_MAX
#define I32_MIN -2147483648
typedef unsigned int uint;


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


// let dotprod_filt [n] (vct: [n]f32) (xs: [n]f32) (ys: [n]f32) : f32 =
//   f32.sum (map3 (\v x y -> x * y * if (f32.isnan v) then 0.0 else 1.0) vct xs ys)
float dotProdFilt(uint n, float* Xvct, float* XTvct, float* yvct) {
    float acc = 0.0;
    for (uint i = 0; i < n; i++) {
        if (yvct[i] != F32_MIN) {
            acc += Xvct[i] * XTvct[i];
        }
    }
    return acc;
}



// let matmul_filt [n][p][m] (xss: [n][p]f32) (yss: [p][m]f32) (vct: [p]f32) : [n][m]f32 =
//   map (\xs -> map (dotprod_filt vct xs) (transpose yss)) xss
// [p][m]
// [3][2]
// XT = [[1,2],
//       [3,4],
//       [5,6]]
// [n][p]
// [2][3]
// X  = [[1,3,5],
//       [2,4,6]]
// y  = [8,9,7]
// xss = Xn, yss = XTn, vct = y
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



void transpose(float* M, float* MT, uint m, uint N) {
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < N; col++) {
            MT[col*m + row] = M[row*N + col];
        }
    }
}



void mkXsqrOptim(uint n, uint N, uint m, float* X, float* XT, float* sample, float* Xsqr, uint K) {
    
    // printf("************** FØR TRANSPOSE ****************\n");
    // for (int i = 0; i < N*m; i++) {
    //     printf("%f, ", sample[i]);
    // }
    // printf("\n\n\n");

    // printf("************** EFTER TRANSPOSE ****************\n");
    // for (int i = 0; i < N*m; i++) {
    //     printf("%f, ", YT[i]);
    // }
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
                                    // if (YT[q*m + ii+i1] != F32_MIN) {
                                    if (sample[(ii+i1)*N + q] != F32_MIN) {
                                        acc[i1] += ab;          // acc[i1] += ab * (1.0-isnan(yqsh[i1]));
                                    }
                                }
                            }

                        }
                        for (int i2 = 0; i2 < R; i2++) { // fully unroll
                            if (ii+i2 < m) {
                                // Xsqr[pix*K*K + i*K + j] = acc;
                                Xsqr[(ii+i2)*(K*K) + j1*K + j2*K] = acc[i2];
                            }
                        }
                    }
                }
            }

}


// ----------
//   let Xinv = intrinsics.opaque <|
//              map mat_inv Xsqr

//   let gauss_jordan [nm] (n:i32) (m:i32) (A: *[nm]f32): [nm]f32 =
//     loop A for i < n do
//       let v1 = A[i]
//       let A' = map (\ind -> let (k, j) = (ind / m, ind % m)
//                             in if v1 == 0.0 then unsafe A[k*m+j] else
//                             let x = unsafe (A[j] / v1) in
//                                 if k < n-1  -- Ap case
//                                 then unsafe ( A[(k+1)*m+j] - A[(k+1)*m+i] * x )
//                                 else x      -- irow case
//                    ) (iota nm)
//       in  scatter A (iota nm) A'


// procedure Naive Gauss(n, (ai j ), (bi ), (xi ))
// integer i, j, k, n; real sum, xmult
// real array (ai j )1:n×1:n , (bi )1:n , (xi )1:n
// for k = 1 to n − 1 do           // row
//     for i = k + 1 to n do       // row we are working on
//         xmult ← aik/akk
//         aik ← xmult
//         for j = k + 1 to n do   // column
//             aij ← aij −(xmult)akj
//         end for
//         bi ← bi − (xmult)bk
//     end for
// end for
// xn ←bn/ann
// for i =n−1to1 step−1do
//     sum ← bi
//     for j = i + 1 to n do
//         sum ← sum − ai j x j
//     end for
// xi ← sum/aii
// end for
// end procedure Naive Gauss

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


//   let mat_inv [n] (A: [n][n]f32): [n][n]f32 =
//     let m = 2*n
//     let nm= n*m
//     -- Pad the matrix with the identity matrix.
//     let Ap = map (\ind -> let (i, j) = (ind / m, ind % m)
//                           in  if j < n then unsafe ( A[i,j] )
//                                        else if j == n+i
//                                             then 1.0
//                                             else 0.0
//                  ) (iota nm)
//     let Ap' = gauss_jordan n m Ap
//     -- Drop the identity matrix at the front!
//     in (unflatten n m Ap')[0:n,n:2*n]

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



// let matvecmul_row_filt [n][m] (xss: [n][m]f32) (ys: [m]f32) =
//     map (\xs -> map2 (\x y -> if (f32.isnan y) then 0 else x*y) xs ys |> f32.sum) xss

void mvMulFilt(uint n, uint N, float* X, float* y, uint K, float* B0){

    for (int i = 0; i < K; i++) {
        uint XIdx  = i*N;

        B0[i] = dotProdFilt(n, &X[XIdx], y, y);
    }
}



// let beta0  = map (matvecmul_row_filt Xh) Yh   -- [2k+2]
//                |> intrinsics.opaque
// let β0 = mvMulFilt X[:,:n] y[:n]
void mkB0(uint m, uint n, uint N, float* X, uint K, float* sample, float* B0){
    for (uint pix = 0; pix < m; pix++) {
        mvMulFilt(n, N, X, &sample[pix*N], K, &B0[pix*K]);
    }
}


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



void mvMul(float* M, float* v, uint rows, uint cols, float* res_v) {

    // for ker6 er rows = N og cols = K
    for (int i = 0; i < rows; i++) {
        float acc = 0.0;

        for (uint elm = 0; elm < cols; elm++) {
            uint MIdx = i*cols + elm;
            acc += M[MIdx] * v[elm];
        }
        res_v[i] = acc;
    }
}


// let β = mvMul Xsqr−1 β0
void ker5(uint m, float* XsqrInvLess, uint K, float* B0, float* B){
    for (uint pix = 0; pix < m; pix++) {
        mvMul(&XsqrInvLess[pix*(K*K)], &B0[pix*K], K, K, &B[pix*K]);
    }
}



void mkB(uint m, float* XsqrInvLess, uint K, float* B0, float* B){
    for (uint pix = 0; pix < m; pix++) {
        for (int i = 0; i < K; i++) {
            float acc = 0.0;

            for (uint j = 0; j < K; j++) {
                acc += XsqrInvLess[pix*(K*K) + i*K + j] * B0[pix*K + j];
            }
            B[pix*K + i] = acc;
        }
    }        
}



// -- yˆ,r,I : [N]f32
// let yˆ = mvMul XT β
void ker6(uint m, uint N, float* XT, float* B, uint K, float* yhat) {
    for (uint pix = 0; pix < m; pix++) {
        mvMul(XT, &B[pix*K], N, K, &yhat[pix*N]);
    }

}



// -- filterPadWithKeys (\y -> !(f32.isnan y)) (f32.nan) y_error_all
// -- Input:   p:(p->value:true or nan:false) dummy:nan arr:[nan,float,nan,float]
// -- Returns: ([(float,int),(float,int)],int)
// let filterPadWithKeys [n] 't
//            (p : (t -> bool))
//            (dummy : t)
//            (arr : [n]t) : ([n](t,i32), i32) =
//   -- [0,1,0,1] <- [nan,float,nan,float]
//   let tfs = map (\a -> if p a then 1 else 0) arr
//   -- number of valid
//   let isT = scan (+) 0 tfs
//   let i   = last isT
//   -- isT:  [0,1,1,2]
//   -- inds: [-1,0,-1,1]
//   let inds= map2 (\a iT -> if p a then iT-1 else -1) arr isT
//   --X [nan,nan,nan,nan]
//   --I inds: [-1,0,-1,1]
//   --D [nan,float,nan,float]
//   --R [float,float,nan,nan]
//   let rs  = scatter (replicate n dummy) inds arr
//   --X [0,0,0,0]
//   --I inds: [-1,0,-1,1]
//   --D [0,1,2,3]
//   --R [1,3,0,0]
//   let ks  = scatter (replicate n 0) inds (iota n)
//   in  (zip rs ks, i)


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

  // let (Nss, y_y_errors, val_indss) = ( intrinsics.opaque <| unzip3 <|
  //   -- y p
  //   map2 (\y y_pred ->
  //           let y_error_all = zip y y_pred |>
  //               map (\(ye,yep) -> if !(f32.isnan ye)
  //                                 then ye-yep else f32.nan )
  //           -- [nan,dif,nan,dif]
  //           let (tups, Ns) = filterPadWithKeys (\y -> !(f32.isnan y)) (f32.nan) y_error_all
  //           -- (tups:([false,],[nan,dif,nan,dif]), Ns:float#ofvalid)
  //           let (y_error, val_inds) = unzip tups
  //           in  (Ns, y_error, val_inds)
  //        ) images y_preds )


// let (N,r,I)= map2 (-) y yˆ |> filterNaNsWKeys
void ker7(uint m, uint N, float* yhat, float* y_errors_all, uint* Nss, float* y_errors, float* sample, int* val_indss) {
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


void MO_fsts_comp(int hmax, int* hs, float* y_errors, uint* nss, float* MO_fsts) {

    for (int i = 0; i < hmax; i++) {

        if (i < *hs) {
            uint idx = i + *nss - *hs + 1;
            *MO_fsts += y_errors[idx];
        }
    }
}


void ker9(uint m, uint N, int* hs, float* y_errors, uint* nss, float* MO_fsts) {
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


void ker10(float lam, uint m, uint n, uint N, float* bound, uint* Nss,
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




