#include <stdio.h>
#include <stdio.h>



// (trend: i32) (k: i32) (n: i32) (freq: f32) (hfrac: f32) (lam: f32) (mappingindices : [N]i32) (images : [m][N]f32)
// (trend, k, m, n, N, freq, hfrac, lam, mappingindices, sample)

int main(int argc, char const *argv[]) {
	
	if (argc != 10) {
		printf("Missing input.\n");
	}

	int trend = argv[1];
	int k = argv[2];
	int m = argv[3];
	int n = argv[4];
	int N = argv[5];
	float freq = argv[6];
	float hfrac = argv[7];
	float lam = argv[8];
	int* mappingindices = malloc(N*sizeof(int));
	mappingindices = argv[9];
	float ** sample = malloc((N*m)*sizeof(float));
	sample = argv[10];

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



