#ifndef KERNELS
#define KERNELS

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





#if 0
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
#endif



__global__ void ker3(uint M, uint K, float* A, float* AI){
    int i = blockIdx.x;
    int k1 = threadIdx.x;
    int k2 = threadIdx.y;

    extern __shared__ float shared[]; // 2*K*K
    float* Ash = &shared[0];
    float* AshTmp = &shared[2*K*K];

    // Ash[k1*K + k2] = A[i*K*K + k1*K + k2] - 1;

    // AI[i*K*K + k1*K + k2] = Ash[k1*K + k2];

    // extern __shared__ float Ash[];
    // float* Ash    = (float*) calloc(2*K*K,sizeof(float));
    // float* AshTmp    = (float*) calloc(2*K*K,sizeof(float));
    // extern __shared__ float AT[];

    // copy the data from the device memory to the first half of the shared mem
    Ash[k1*K + k2]     = A[i*K*K + k1*K + k2];
    // writes the identity matrix to the second half
    Ash[k1*2*K + K + k2] = (float) (k2 == k1);

    #pragma unroll
    for (uint q = 0; q < 2*K; q++){               // sequential
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
        __syncthreads();
        AshTmp[k1*2*K + k2] = tmp;

        // barrier for block-level sync
        __syncthreads();

        // swap pointers
        float* tmp2 = AshTmp;
        AshTmp = Ash;
        Ash    = tmp2;
    }

    AI[i*K*K + k1*K + k2] = Ash[k1*2*K + K + k2];

}

#if 0
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
                    // barrier for block-level sync
                    AshTmp[k1*2*K + k2] = tmp;
                    // barrier for block-level sync
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
    }
}
#endif




#endif


