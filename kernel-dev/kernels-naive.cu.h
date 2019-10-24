#ifndef KERNELS
#define KERNELS

#define BLOCK_SIZE 1024//1024 //1024//2048
#define WIDTH_A  1024//1024 //1024//2048
#define HEIGHT_A 1//2048//2048//2048
#define WIDTH_B  1024//4096//2048
#define TILE_HEIGHT 1
#define TILE_WIDTH 1024

#define PI 3.14159265
#define F32_MIN -FLT_MAX
#define I32_MIN -2147483648

typedef unsigned int uint;

__global__ void ker1(uint N, int K, int freq, int* mappingindices, float* X, float* XT){

	int gid = blockIdx.x*blockDim.x + threadIdx.x;
	int cols = gid % N;
	int rows = gid / N;
	int colsT = cols * K + rows;
	int rowsT = cols % K;

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


// Kernel 3
__global__ void ker3(uint m, uint K, float* Xsqr, float* XsqrInv, float* d_XsqrInvLess){
	int gid = blockIdx.x*blockDim.x + threadIdx.x;

    float* XsqrPix = &Xsqr[gid*K*K];
    float* XsqrInvPix = &XsqrInv[gid*K*K*2];
    float* d_XsqrInvLessPix = &d_XsqrInvLess[gid*K*K];

    uint cols = 2*K;
    uint identIdx = K*cols;

    // copy data
    for (uint i = 0; i < K; i++){
        for (uint j = 0; j < K; j++){
            uint sqrIdx = i*K + j;
            uint invIdx = i*cols + j;
            XsqrInvPix[invIdx] = XsqrPix[sqrIdx];
        }
    }

    // appending the identity matrix
    for (uint i = 0; i < K; i++){
        for (uint j = K; j < identIdx; j+=cols+1){
            uint idx = i*identIdx + j;
            XsqrInvPix[idx] = 1.0;
        }
    }


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
            XsqrInvPix[i*cols+j] = XsqrInvPix[i*cols+j] / temp;
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

    // Writing the second half of the 2*K*K matrix to output
    for (int pix = 0; pix < m; pix++) {
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                uint XinvIdx  = pix*K*(2*K) + i*(K*2) + j+K;
                uint XlessIdx = pix*K*K + i*K + j;
                d_XsqrInvLessPix[XlessIdx] = XsqrInvPix[XinvIdx];
            }
        }
    }

}






// Kernel 4
__global__ void ker4(uint m, uint n, uint N, float* X, uint K, float* sample, float* B0){

    int pix = blockIdx.x;
    int i = threadIdx.y;
    float accum = 0.0f;

    for(int k = 0; k < n; k++) {
        // setting valid bit of valid pixel data
        float cur_y = sample[pix*N+k];

        if (cur_y == F32_MIN) {
            accum += 0.0;
        } else {
            accum += X[i*N+k] * cur_y;
        }
    }

    // adding results to beta0
    B0[pix*K + i] = accum;
}


#endif


