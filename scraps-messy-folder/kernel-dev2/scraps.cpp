
// void mkXsqrOptim(uint n, uint N, uint m, float* X, float* XT, float* sample, float* Xsqr, uint K) {
    
//     float* YT = (float*) calloc(N*m,sizeof(float));
//     transposeMatrix(sample, YT, m, N);

//     const int R = 30;

//     for (int ii = 0; ii < m; ii+=R) {                                 // forall, grid.z
//         for (int jj1 = 0; jj1 < K; jj1+=K) {                          // forall, grid.y
//             for (int jj2 = 0; jj2 < K; jj2+=K){                       // forall, grid.x
//                 for (int j1 = jj1; j1 < min(jj1+K, K); j1++) {        // forall, block.y
//                     for (int j2 = jj2; j2 < min(jj2+K, K); j2++) {    // forall, block.x

//                         // float yqsh[R];          // size R, shared memory
//                         float acc[R]; //  = calloc(R,sizeof(float));          // size R, registers

//                         for (int i = 0; i < R; i++) {                   // fully unroll
//                             acc[i] = 0.0;
//                         }
//                         float a, b, ab; //, y; 
//                         for (int q = 0; q < n; q++) {
//                             a = X[j1*N + q];       // a = X[j1, q];
//                             b = XT[q*K + j2];      // b = XT[q, j2];
//                             ab = a*b;
//                             // collective copy global-to-shared
//                             // for (int idx = 0; idx < R; idx++) {
//                             //     yqsh[idx] = YT[q, ii]; // YT[q,ii:min(ii+R,M)] ?????????????
//                             // }
//                             // barrier; // block-level synch         

//                             for (int i1 = 0; i1 < R; i1++) { // fully unroll
//                                 if (ii+i1 < m) {
//                                     if (YT[q*m + (ii+i1)] != F32_MIN) {
//                                     // if (sample[(ii+i1)*N + q] != F32_MIN) {
//                                         acc[i1] += ab;          // acc[i1] += ab * (1.0-isnan(yqsh[i1]));
//                                     }
//                                 }
//                             }

//                             for (int i2 = 0; i2 < R; i2++) { // fully unroll
//                                 if ((ii+i2) < m) {
//                                     // Xsqr[pix*K*K + i*K + j] = acc;
//                                     Xsqr[(ii+i2)*(K*K) + j1*K + j2] = acc[i2];
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }

// }



for (int ii = 0; ii < m; ii+=R) {                                 // forall, grid.z
        for (int jj1 = 0; jj1 < K; jj1+=K) {                          // forall, grid.y
            for (int jj2 = 0; jj2 < K; jj2+=K){                       // forall, grid.x
                for (int j1 = jj1; j1 < min(jj1+K, K); j1++) {        // forall, block.y
                    for (int j2 = jj2; j2 < min(jj2+K, K); j2++) {    // forall, block.x

                        // float yqsh[R];          // size R, shared memory
                        float acc[R]; //  = calloc(R,sizeof(float));          // size R, registers

                        #pragma unroll
                        for (int i = 0; i < R; i++) {                   // fully unroll
                            acc[i] = 0.0;
                        }
                        float a, b, ab; 
                        for (int q = 0; q < n; q++) {
                            a = X[j1*N + q];       // a = X[j1, q];
                            b = XT[q*K + j2];      // b = XT[q, j2];
                            ab = a*b;
                            // collective copy global-to-shared
                            // for (int idx = 0; idx < R; idx++) {
                            //     yqsh[idx] = YT[q, ii]; // YT[q,ii:min(ii+R,M)] ?????????????
                            // }
                            // barrier; // block-level synch         

                            #pragma unroll
                            for (int i1 = 0; i1 < R; i1++) { // fully unroll
                                if (ii+i1 < m) {
                                    if (YT[q*m + (ii+i1)] != F32_MIN) {
                                    // if (sample[(ii+i1)*N + q] != F32_MIN) {
                                        acc[i1] += ab;          // acc[i1] += ab * (1.0-isnan(yqsh[i1]));
                                    }
                                }
                            }

                            #pragma unroll
                            for (int i2 = 0; i2 < R; i2++) { // fully unroll
                                if ((ii+i2) < m) {
                                    // Xsqr[pix*K*K + i*K + j] = acc;
                                    Xsqr[(ii+i2)*(K*K) + j1*K + j2] = acc[i2];
                                }
                            }
                        }
                    }
                }
            }
        }
    }


