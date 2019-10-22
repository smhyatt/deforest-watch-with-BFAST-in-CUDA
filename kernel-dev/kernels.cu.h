#ifndef KERNELS
#define KERNELS

#define PI 3.14159265

typedef unsigned int uint;

__global__ void ker1(uint N, int K, int freq, int* mappingindices, float* X){

	int gid = blockIdx.x*blockDim.x + threadIdx.x;
	int cols = gid % N;
	int rows = gid / N;

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
	}


    // for (uint i = 0; i < K; i++){
    //     for (uint j = 0; j < N; j++){
    //         float ind = mappingindices[j];
    //         uint index = i*N + j;

    //         if(i==0){
    //             X[index] = 1.0;
    //         } else if(i==1){
    //             X[index] = ind;
    //         } else {
    //             float ip = (float)(i / 2);
    //             float jp = ind;
    //             float angle = 2 * PI * ip * jp / f;
    //             if(i%2 == 0) {
    //                 X[index] = sin(angle);
    //             } else {
    //                 X[index] = cos(angle);
    //             }
    //         }
    //     }
    // }
}


// void transpose(uint N, int K, float* X, float* XT) {
//     for (uint i = 0; i < K; i++){
//         for (uint j = 0; j < N; j++){
//             uint Xidx  = i*N + j;
//             uint XTidx = j*K + i;
//             XT[XTidx]  = X[Xidx];
//         }
//     }
// }


#endif


