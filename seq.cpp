#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>



// (trend: i32) (k: i32) (n: i32) (freq: f32) (hfrac: f32) (lam: f32) (mappingindices : [N]i32) (images : [m][N]f32)
// (trend, k, m, n, N, freq, hfrac, lam, mappingindices, sample)

int main(int argc, char const *argv[]) {
	
	if (argc != 10) {
		std::cerr << "Not enough arguments!\n";
      	return -1;
	}

	int trend 			= atoi(argv[1]);
	int k 				= atoi(argv[2]);
	int m 				= atoi(argv[3]);
	int n 				= atoi(argv[4]);
	int N 				= atoi(argv[5]);
	float freq 			= atoi(argv[6]);
	float hfrac 		= atoi(argv[7]);
	float lam 			= atoi(argv[8]);
	int size 			= N*sizeof(int);
	int* mappingindices = (int*)malloc(size);
	for (int i = 0; i < N; i++) {
		mappingindices[i] = argv[9][i];
	}

	int sampsize 		= (N*m)*sizeof(float);
	float ** sample 	= (float**)malloc(sampsize);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < N; j++) {
			sample[i][j] = argv[10][i+j+sizeof(float)];
		}
	}


	printf("%A\n", trend);
	printf("%A\n", k);
	printf("%A\n", m);
	printf("%A\n", n);
	printf("%A\n", N);
	printf("%A\n", freq);
	printf("%A\n", hfrac);
	printf("%A\n", lam);
	printf("%A\n", mappingindices);
	printf("%A\n", sample);

	return 0;
}



