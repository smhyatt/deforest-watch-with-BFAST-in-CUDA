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
	int idxT  = rowsT * K + colsT;

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




#endif


