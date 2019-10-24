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

        if(rows == 0){
            X[gid] = 1.0;
        } else if(rows == 1){
            X[gid] = mi_val;
        } else {
            float angle = 2 * PI * ((float)(rows / 2)) * (float)mi_val / freq;
            if(rows % 2 == 0) {
                X[gid] = sin(angle);
            } else {
                X[gid] = cos(angle);
            }
        }

        XT[idxT]  = X[gid];
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

// ker2 <<< grid, block >>> (n, N, m, d_X, d_XT, d_sample, d_Xsqr, K);

__global__ void ker2(uint n, uint N, uint m, float* X, float* XT, float* sample,
                     float* Xsqr, uint K) {

    int pix = blockIdx.x; 
    int i = threadIdx.y;
    int j = threadIdx.x;
    float accum = 0.0f;

    for(int k = 0; k < n; k++) {
        int valid = !(sample[pix*N+k] == F32_MIN);
        printf("%d\n", valid);
        accum += X[i*N+k] * XT[k*K+j] * valid;
    }

    Xsqr[pix*K*K + i*K + j] = accum;
}




__global__ void ker4(uint m, uint n, uint N, float* X, uint K, float* sample, float* B0){
    float accum = 0.0f;

    // setting global thread 
    int gidx = blockIdx.x*blockDim.x + threadIdx.x;

    // defining thread limit
    if( gidx >= K*m ) return;

    for(int i = 0; i < n; i ++) {
        // setting valid bit of valid pixel data
        int valid = !(sample[blockIdx.x*N+i] == F32_MIN);
        accum += X[gidx*N + i] * valid;
    }

    // adding results to beta0 
    B0[gidx*K] = accum;
}



#endif


