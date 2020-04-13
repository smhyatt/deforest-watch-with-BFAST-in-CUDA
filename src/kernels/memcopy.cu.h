#ifndef MEMCOPY
#define MEMCOPY


typedef unsigned int uint;


/**
 * Naive memcpy kernel, for the purpose of comparing with
 * a more "realistic" bandwidth number.
 */
__global__ void naiveMemcpyF32(float* d_out, float* d_inp, const uint N) {
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < N) {
        d_out[gid] = d_inp[gid];
    }
}


__global__ void naiveMemcpyI32(int* d_out, int* d_inp, const uint N) {
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < N) {
        d_out[gid] = d_inp[gid];
    }
}


__global__ void naiveMemcpyUI32(uint* d_out, uint* d_inp, const uint N) {
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < N) {
        d_out[gid] = d_inp[gid];
    }
}


#endif

