#ifndef KERNELS
#define KERNELS

#define PI 3.14159265

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



float dotProdFilt(uint n, float* Xvct, float* XTvct, float* yvct) {
    float acc = 0.0;
    for (uint i = 0; i < n; i++) {
        if (yvct[i] != F32_MIN) {
            acc += Xvct[i] * XTvct[i];
        }
    }
    return acc;
}



void mmMulFilt(uint n, uint N, float* X, float* XT, float* y, float* Xsqr, uint K, float* tspVct){

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            uint XIdx = i*N;
            uint XTIdx = j;
            uint resIdx = i*K + j;

            for (uint l = 0; l < n; l++) {
                uint idx  = l*K + j;
                tspVct[l] = XT[idx];
            }

            Xsqr[resIdx] = dotProdFilt(n, &X[XIdx], tspVct, y);
        }
    }
}

void ker2(uint n, uint N, uint m, float* X, float* XT, float* sample, float* Xsqr, uint K) {
	float* tspVct = calloc(n,sizeof(float));

    for (uint pix = 0; pix < m; pix++) {
        mmMulFilt(n, N, X, XT, &sample[pix*N], &Xsqr[pix*K*K], K, tspVct);
    }

    free(tspVct);
}




#endif


