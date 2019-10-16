#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265
typedef unsigned int uint;

// (trend: i32) (k: i32) (n: i32) (freq: f32) (hfrac: f32) (lam: f32) (mappingindices : [N]i32) (images : [m][N]f32)
// (trend, k, m, n, N, freq, hfrac, lam, mappingindices, sample)
int trend 			= 1;
int k 				= 3;
int m 				= 2;
int n 				= 113;
int N 				= 235;
float freq 			= 365.000000;
float hfrac 		= 0.250000;
float lam 			= 1.736126;

int mappingindices[235] = {122, 138, 170, 202, 218, 250, 266, 314, 330, 378, 394, 474, 490, 570, 586, 602, 618, 666, 730, 1290, 1306, 1322, 1338, 1354, 1370, 1386, 1402, 1482, 1594, 1642, 1674, 1706, 1722, 1802, 1818, 1850, 1898, 1994, 2010, 2042, 2074, 2146, 2154, 2194, 2234, 2274, 2322, 2330, 2338, 2346, 2378, 2410, 2418, 2426, 2442, 2450, 2458, 2474, 2490, 2586, 2594, 2650, 2658, 2682, 2690, 2698, 2706, 2722, 2730, 2738, 2754, 2762, 2770, 2778, 2794, 2810, 2818, 2826, 2834, 2898, 2930, 3034, 3050, 3058, 3066, 3074, 3082, 3106, 3122, 3130, 3146, 3154, 3178, 3194, 3242, 3266, 3274, 3354, 3370, 3418, 3426, 3458, 3466, 3498, 3514, 3522, 3530, 3538, 3562, 3602, 3610, 3626, 3650, 3778, 3786, 3810, 3818, 3874, 3882, 3890, 3906, 3922, 3938, 3946, 3962, 3970, 3994, 4002, 4090, 4138, 4210, 4218, 4242, 4250, 4258, 4266, 4274, 4282, 4290, 4306, 4354, 4434, 4466, 4482, 4514, 4530, 4546, 4562, 4578, 4594, 4610, 4626, 4642, 4658, 4722, 4738, 4858, 4866, 4874, 4890, 4922, 4994, 5002, 5018, 5026, 5042, 5050, 5058, 5082, 5106, 5114, 5130, 5138, 5162, 5202, 5210, 5234, 5250, 5282, 5290, 5298, 5306, 5322, 5362, 5378, 5386, 5410, 5418, 5426, 5442, 5450, 5458, 5466, 5514, 5530, 5562, 5618, 5626, 5642, 5650, 5658, 5666, 5682, 5690, 5698, 5706, 5714, 5730, 5738, 5746, 5754, 5762, 5786, 5802, 5810, 5818, 5858, 5866, 5874, 5914, 5946, 5970, 5978, 5994, 6010, 6026, 6034, 6042, 6050, 6066, 6074, 6082, 6090, 6098, 6114};
float sample[2][235] = {{-10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4716.000000, -10000.000000, 4147.000000, 4546.000000, -10000.000000, 3565.000000, -10000.000000, -10000.000000, -10000.000000, 4208.000000, 4446.000000, -10000.000000, 3653.000000, 3734.000000, 4064.000000, -10000.000000, -10000.000000, 4628.000000, 4034.000000, -10000.000000, 3867.000000, 2117.000000, 4598.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4210.000000, -10000.000000, 5253.000000, 4219.000000, -10000.000000, 3074.000000, -10000.000000, -10000.000000, 3982.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 3888.000000, -10000.000000, -10000.000000, -10000.000000, 3662.000000, 4182.000000, 4475.000000, 4659.000000, 5102.000000, 4552.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 3827.000000, -10000.000000, -10000.000000, -10000.000000, 3773.000000, 3985.000000, -10000.000000, -10000.000000, 5331.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4165.000000, -10000.000000, 3965.000000, -10000.000000, 4094.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 3778.000000, -10000.000000, 3176.000000, 4073.000000, 4461.000000, 4017.000000, 4031.000000, 3929.000000, 3999.000000, -10000.000000, 4349.000000, 4182.000000, -10000.000000, -10000.000000, -10000.000000, 3001.000000, 3286.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4212.000000, 4309.000000, -10000.000000, 4451.000000, -10000.000000, -10000.000000, 3999.000000, 4064.000000, -10000.000000, -10000.000000, 4215.000000, -10000.000000, -10000.000000, -10000.000000, 3917.000000, 4035.000000, 4263.000000, 4286.000000, -10000.000000, -10000.000000, 4062.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4129.000000, 4042.000000, -10000.000000, 3923.000000, -10000.000000, 4407.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4716.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4690.000000, 4570.000000, 4505.000000, -10000.000000, 4172.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4225.000000, -10000.000000, -10000.000000, -10000.000000, 4402.000000, 4168.000000, 4053.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4237.000000, -10000.000000, -10000.000000, -10000.000000, 4035.000000, 3533.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4050.000000, 4341.000000, -10000.000000, 4555.000000, 4096.000000, -10000.000000, 4077.000000, 4196.000000, -10000.000000, -10000.000000, 4205.000000, 4647.000000, -10000.000000, 4005.000000, 3607.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 3977.000000, 3761.000000, -10000.000000, -10000.000000, -10000.000000, 4184.000000, 2652.000000, 4341.000000, -10000.000000, -10000.000000, 4476.000000, 4257.000000, -10000.000000, 4066.000000, -10000.000000, -10000.000000, -10000.000000, 4836.000000, -10000.000000, -10000.000000, 4048.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4533.000000, 4330.000000, -10000.000000, 4554.000000, 4100.000000, -10000.000000, 3692.000000, -10000.000000, 4337.000000, -10000.000000, -10000.000000}, {-10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4581.000000, -10000.000000, 4318.000000, 4602.000000, -10000.000000, 3480.000000, -10000.000000, -10000.000000, -10000.000000, 4236.000000, 3872.000000, -10000.000000, 4249.000000, 4039.000000, 3906.000000, -10000.000000, -10000.000000, 4074.000000, 3956.000000, -10000.000000, 4107.000000, 1820.000000, 3989.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4152.000000, -10000.000000, 4480.000000, 3969.000000, -10000.000000, 3703.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 3888.000000, -10000.000000, -10000.000000, -10000.000000, 3728.000000, 3905.000000, 4534.000000, 4337.000000, 4563.000000, 4268.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 3717.000000, -10000.000000, -10000.000000, -10000.000000, 3557.000000, 3607.000000, -10000.000000, -10000.000000, 4753.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4432.000000, -10000.000000, 4251.000000, -10000.000000, 4159.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 3535.000000, -10000.000000, 3275.000000, 4024.000000, 4886.000000, 4409.000000, 4170.000000, 4220.000000, 4211.000000, -10000.000000, 4320.000000, 4102.000000, -10000.000000, -10000.000000, -10000.000000, 3058.000000, 3678.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4426.000000, 4250.000000, -10000.000000, 4244.000000, -10000.000000, -10000.000000, 3995.000000, 3987.000000, -10000.000000, -10000.000000, 4078.000000, -10000.000000, -10000.000000, -10000.000000, 4093.000000, 4165.000000, 4322.000000, 4239.000000, -10000.000000, -10000.000000, 3718.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4234.000000, 3894.000000, -10000.000000, 3684.000000, -10000.000000, 4341.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4805.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4369.000000, 4538.000000, 4351.000000, -10000.000000, 4270.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4115.000000, -10000.000000, -10000.000000, -10000.000000, 4373.000000, 4262.000000, 4005.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4261.000000, -10000.000000, -10000.000000, -10000.000000, 4027.000000, 3533.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 3866.000000, 4232.000000, -10000.000000, 4513.000000, 4038.000000, -10000.000000, 4164.000000, 4045.000000, 3489.000000, -10000.000000, 3896.000000, 4625.000000, -10000.000000, 4224.000000, 3571.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 3719.000000, 3361.000000, -10000.000000, -10000.000000, -10000.000000, 4295.000000, 2625.000000, 4151.000000, -10000.000000, -10000.000000, 4310.000000, 4059.000000, -10000.000000, 4152.000000, -10000.000000, -10000.000000, -10000.000000, 4584.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4334.000000, 4289.000000, -10000.000000, 4303.000000, 3995.000000, -10000.000000, 3847.000000, -10000.000000, 4242.000000, -10000.000000, -10000.000000}};


void ker1(int kp, int f, float* X){
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


void transpose(int K, float* X, float* XT) {
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
float dotProdFilt(float* Xvct, float* XTvct, float* yvct) {
    float acc = 0.0;
    for (uint i = 0; i < n; i++) {
        if (yvct[i] != -10000.000000) {
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
void mmMulFilt(float* X, float* XT, float* y, float* Xsqr, uint K){
    float* tspVct = calloc(n,sizeof(float));

    // K
    for (int i = 0; i < K; i++) {
        // K
        for (int j = 0; j < K; j++) {
            uint XIdx = i*N;
            uint XTIdx = j;
            uint resIdx = i*K + j;

            for (uint l = 0; l < n; l++) {
                uint idx  = l*K + j;
                tspVct[l] = XT[idx];
            }

            Xsqr[resIdx] = dotProdFilt(&X[XIdx], tspVct, y);
        }
    }

    // free(tspVct);
}


// -- Xsqr,Xsqr−1:[K][K]f32; β0,β:[K]f32
// let Xsqr = mmMulFilt X[:,:n] XT[:n,:] y[:n] -- ker 2
void ker2(float* X, float* XT, float* Xsqr, uint K) {
    uint numPix = sizeof(sample)/sizeof(sample[0]);

    for (uint pix = 0; pix < numPix; pix++) {
        mmMulFilt(X, XT, &sample[pix][0], Xsqr, K);
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


void doubleDown(float* XsqrInv, float* XsqrInvLess, uint K) {
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            uint XinvIdx  = i*(K*2) + j+K; // XsqrInv er K længere
            uint XlessIdx = i*K + j;
            XsqrInvLess[XlessIdx] = XsqrInv[XinvIdx];
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

void ker3(float* Xsqr, float* XsqrInv, uint K){
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

    gaussJordan(XsqrInv, cols, K);

}



// let matvecmul_row_filt [n][m] (xss: [n][m]f32) (ys: [m]f32) =
//     map (\xs -> map2 (\x y -> if (f32.isnan y) then 0 else x*y) xs ys |> f32.sum) xss

void mvMulFilt(float* X, float* y, uint K, float* B0){

    for (int i = 0; i < K; i++) {
        uint XIdx = i*N;

        B0[i] = dotProdFilt(&X[XIdx], y, y);
    }
}



// let beta0  = map (matvecmul_row_filt Xh) Yh   -- [2k+2]
//                |> intrinsics.opaque
// let β0 = mvMulFilt X[:,:n] y[:n]
void ker4(float* X, uint K, float* B0){
    uint numPix = sizeof(sample)/sizeof(sample[0]);

    for (uint pix = 0; pix < numPix; pix++) {
        mvMulFilt(X, &sample[pix][0], K, B0);
    }
}



void mvMul(float* M, float* v, uint rows, uint cols, float* res_v) {

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
void ker5(float* XsqrInv, uint K, float* B0, float* B){
    mvMul(XsqrInv, B0, K, K, B);
}


// -- yˆ,r,I : [N]f32
// let yˆ = mvMul XT β
void ker6(float* XT, float* B, uint K, float* yhat) {
    mvMul(XT, B, N, K, yhat);
}


let (N,r,I)= map2 (-) y yˆ |> filterNaNsWKeys
void ker7() {

}

int main(int argc, char const *argv[]) {

	if (argc > 1) {
		printf("No arguments!\n");
      	return -1;
	}

    uint mLen = sizeof(sample)/sizeof(sample[0]);

	printf("%d\n", trend);
	printf("%d\n", k);
	printf("%d\n", m);
	printf("%d\n", n);
	printf("%d\n", N);
	printf("%f\n", freq);
	printf("%f\n", hfrac);
	printf("%f\n", lam);
	printf("%lu\n", sizeof(mappingindices)/sizeof(mappingindices[0]));
	printf("%u\n", mLen);

    int K = 2*k +2;
    float* X  = calloc(K*N,sizeof(float));
    float* XT = calloc(K*N,sizeof(float));
    ker1(K,freq,X);
    transpose(K,X,XT);

    printf("\n****** Printing X ******\n");
    for (size_t i = 0; i < 1; i++){ // i < K
        for (size_t j = 0; j < n; j++){
            uint index = i*N + j;
            printf(" %f ", X[index]);
        }
        printf("\n");
    }

    // printf("\n****** Printing Y ******\n");
    // uint Ylen = sizeof(sample)/sizeof(sample[0]);
    // for (size_t i = 0; i < Ylen; i++){ // i < Ylen
    //     for (size_t j = 0; j < n; j++){
    //         printf(" %lf ", sample[i][j]);
    //     }
    //     printf("\n");
    // }    

    // printf("\n****** Printing XT ******\n");
    // for (size_t i = 0; i < N; i++){
    //     for (size_t j = 0; j < K; j++){
    //         uint index = i*K + j;
    //         printf(" %f ", XT[index]);
    //     }
    //     printf("\n");
    // }

    // [n][m]
    float* Xsqr = calloc(K*K,sizeof(float));
    ker2(X, XT, Xsqr, K);

    printf("\n****** Printing Xsqr ******\n");
    for (size_t i = 0; i < K; i++){
        for (size_t j = 0; j < K; j++){
            uint index = i*K + j;
            printf("%f, ", Xsqr[index]);
        }
        printf("\n");
    }
    printf("\n");

    float* XsqrInv = calloc(2*K*K,sizeof(float));
    ker3(Xsqr,XsqrInv,K);

    printf("\n****** Printing XsqrInv ******\n");
    for (uint i = 0; i < K; i++){
        for (uint j = 0; j < 2*K; j++){
            uint index = i*2*K + j;
            printf("%f, ", XsqrInv[index]);
        }
        printf("\n");
    }
    printf("\n");


    float* XsqrInvLess = calloc(K*K,sizeof(float));
    doubleDown(XsqrInv, XsqrInvLess, K);

    printf("\n****** Printing XsqrInvLess ******\n");
    for (size_t i = 0; i < K; i++){
        for (size_t j = 0; j < K; j++){
            uint index = i*K + j;
            printf("%f, ", XsqrInvLess[index]);
        }
        printf("\n");
    }
    printf("\n");

    float* B0 = calloc(K,sizeof(float));
    ker4(X, K, B0);

    printf("\n****** Printing B0 ******\n");
    for (uint i = 0; i < K; i++){
        printf("%f, ", B0[i]);
    }
    printf("\n");

    float* B = calloc(K,sizeof(float));
    ker5(XsqrInv, K, B0, B);

    printf("\n****** Printing B ******\n");
    for (uint i = 0; i < K; i++){
        printf("%f, ", B[i]);
    }
    printf("\n");

    float* yhat = calloc(N,sizeof(float));
    ker6(XT, B, K, yhat);

    printf("\n****** Printing yhat ******\n");
    for (uint i = 0; i < N; i++){
        printf("%f, ", yhat[i]);
    }
    printf("\n");


    free(X);
    free(XT);
    free(Xsqr);
    free(XsqrInv);
    free(B0);
    free(B);

	return 0;
}



