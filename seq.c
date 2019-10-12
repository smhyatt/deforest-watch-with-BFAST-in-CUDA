#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// #include <string.h>
// #include <iostream>
#define PI 3.14159265
typedef unsigned int uint;

// (trend: i32) (k: i32) (n: i32) (freq: f32) (hfrac: f32) (lam: f32) (mappingindices : [N]i32) (images : [m][N]f32)
// (trend, k, m, n, N, freq, hfrac, lam, mappingindices, sample)
int trend 			= 1;
int k 				= 3;
int m 				= 2;
int n 				= 113;
int N 				= 235;
float freq 			= 365.000000;
float hfrac 		= 0.250000;
float lam 			= 1.736126;

int mappingindices[235] = {122, 138, 170, 202, 218, 250, 266, 314, 330, 378, 394, 474, 490, 570, 586, 602, 618, 666, 730, 1290, 1306, 1322, 1338, 1354, 1370, 1386, 1402, 1482, 1594, 1642, 1674, 1706, 1722, 1802, 1818, 1850, 1898, 1994, 2010, 2042, 2074, 2146, 2154, 2194, 2234, 2274, 2322, 2330, 2338, 2346, 2378, 2410, 2418, 2426, 2442, 2450, 2458, 2474, 2490, 2586, 2594, 2650, 2658, 2682, 2690, 2698, 2706, 2722, 2730, 2738, 2754, 2762, 2770, 2778, 2794, 2810, 2818, 2826, 2834, 2898, 2930, 3034, 3050, 3058, 3066, 3074, 3082, 3106, 3122, 3130, 3146, 3154, 3178, 3194, 3242, 3266, 3274, 3354, 3370, 3418, 3426, 3458, 3466, 3498, 3514, 3522, 3530, 3538, 3562, 3602, 3610, 3626, 3650, 3778, 3786, 3810, 3818, 3874, 3882, 3890, 3906, 3922, 3938, 3946, 3962, 3970, 3994, 4002, 4090, 4138, 4210, 4218, 4242, 4250, 4258, 4266, 4274, 4282, 4290, 4306, 4354, 4434, 4466, 4482, 4514, 4530, 4546, 4562, 4578, 4594, 4610, 4626, 4642, 4658, 4722, 4738, 4858, 4866, 4874, 4890, 4922, 4994, 5002, 5018, 5026, 5042, 5050, 5058, 5082, 5106, 5114, 5130, 5138, 5162, 5202, 5210, 5234, 5250, 5282, 5290, 5298, 5306, 5322, 5362, 5378, 5386, 5410, 5418, 5426, 5442, 5450, 5458, 5466, 5514, 5530, 5562, 5618, 5626, 5642, 5650, 5658, 5666, 5682, 5690, 5698, 5706, 5714, 5730, 5738, 5746, 5754, 5762, 5786, 5802, 5810, 5818, 5858, 5866, 5874, 5914, 5946, 5970, 5978, 5994, 6010, 6026, 6034, 6042, 6050, 6066, 6074, 6082, 6090, 6098, 6114};


float sample[2][235] = {{-10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4716.000000, -10000.000000, 4147.000000, 4546.000000, -10000.000000, 3565.000000, -10000.000000, -10000.000000, -10000.000000, 4208.000000, 4446.000000, -10000.000000, 3653.000000, 3734.000000, 4064.000000, -10000.000000, -10000.000000, 4628.000000, 4034.000000, -10000.000000, 3867.000000, 2117.000000, 4598.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4210.000000, -10000.000000, 5253.000000, 4219.000000, -10000.000000, 3074.000000, -10000.000000, -10000.000000, 3982.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 3888.000000, -10000.000000, -10000.000000, -10000.000000, 3662.000000, 4182.000000, 4475.000000, 4659.000000, 5102.000000, 4552.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 3827.000000, -10000.000000, -10000.000000, -10000.000000, 3773.000000, 3985.000000, -10000.000000, -10000.000000, 5331.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4165.000000, -10000.000000, 3965.000000, -10000.000000, 4094.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 3778.000000, -10000.000000, 3176.000000, 4073.000000, 4461.000000, 4017.000000, 4031.000000, 3929.000000, 3999.000000, -10000.000000, 4349.000000, 4182.000000, -10000.000000, -10000.000000, -10000.000000, 3001.000000, 3286.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4212.000000, 4309.000000, -10000.000000, 4451.000000, -10000.000000, -10000.000000, 3999.000000, 4064.000000, -10000.000000, -10000.000000, 4215.000000, -10000.000000, -10000.000000, -10000.000000, 3917.000000, 4035.000000, 4263.000000, 4286.000000, -10000.000000, -10000.000000, 4062.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4129.000000, 4042.000000, -10000.000000, 3923.000000, -10000.000000, 4407.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4716.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4690.000000, 4570.000000, 4505.000000, -10000.000000, 4172.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4225.000000, -10000.000000, -10000.000000, -10000.000000, 4402.000000, 4168.000000, 4053.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4237.000000, -10000.000000, -10000.000000, -10000.000000, 4035.000000, 3533.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4050.000000, 4341.000000, -10000.000000, 4555.000000, 4096.000000, -10000.000000, 4077.000000, 4196.000000, -10000.000000, -10000.000000, 4205.000000, 4647.000000, -10000.000000, 4005.000000, 3607.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 3977.000000, 3761.000000, -10000.000000, -10000.000000, -10000.000000, 4184.000000, 2652.000000, 4341.000000, -10000.000000, -10000.000000, 4476.000000, 4257.000000, -10000.000000, 4066.000000, -10000.000000, -10000.000000, -10000.000000, 4836.000000, -10000.000000, -10000.000000, 4048.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4533.000000, 4330.000000, -10000.000000, 4554.000000, 4100.000000, -10000.000000, 3692.000000, -10000.000000, 4337.000000, -10000.000000, -10000.000000}, {-10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4581.000000, -10000.000000, 4318.000000, 4602.000000, -10000.000000, 3480.000000, -10000.000000, -10000.000000, -10000.000000, 4236.000000, 3872.000000, -10000.000000, 4249.000000, 4039.000000, 3906.000000, -10000.000000, -10000.000000, 4074.000000, 3956.000000, -10000.000000, 4107.000000, 1820.000000, 3989.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4152.000000, -10000.000000, 4480.000000, 3969.000000, -10000.000000, 3703.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 3888.000000, -10000.000000, -10000.000000, -10000.000000, 3728.000000, 3905.000000, 4534.000000, 4337.000000, 4563.000000, 4268.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 3717.000000, -10000.000000, -10000.000000, -10000.000000, 3557.000000, 3607.000000, -10000.000000, -10000.000000, 4753.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4432.000000, -10000.000000, 4251.000000, -10000.000000, 4159.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 3535.000000, -10000.000000, 3275.000000, 4024.000000, 4886.000000, 4409.000000, 4170.000000, 4220.000000, 4211.000000, -10000.000000, 4320.000000, 4102.000000, -10000.000000, -10000.000000, -10000.000000, 3058.000000, 3678.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4426.000000, 4250.000000, -10000.000000, 4244.000000, -10000.000000, -10000.000000, 3995.000000, 3987.000000, -10000.000000, -10000.000000, 4078.000000, -10000.000000, -10000.000000, -10000.000000, 4093.000000, 4165.000000, 4322.000000, 4239.000000, -10000.000000, -10000.000000, 3718.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4234.000000, 3894.000000, -10000.000000, 3684.000000, -10000.000000, 4341.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4805.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4369.000000, 4538.000000, 4351.000000, -10000.000000, 4270.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4115.000000, -10000.000000, -10000.000000, -10000.000000, 4373.000000, 4262.000000, 4005.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4261.000000, -10000.000000, -10000.000000, -10000.000000, 4027.000000, 3533.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 3866.000000, 4232.000000, -10000.000000, 4513.000000, 4038.000000, -10000.000000, 4164.000000, 4045.000000, 3489.000000, -10000.000000, 3896.000000, 4625.000000, -10000.000000, 4224.000000, 3571.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 3719.000000, 3361.000000, -10000.000000, -10000.000000, -10000.000000, 4295.000000, 2625.000000, 4151.000000, -10000.000000, -10000.000000, 4310.000000, 4059.000000, -10000.000000, 4152.000000, -10000.000000, -10000.000000, -10000.000000, 4584.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, -10000.000000, 4334.000000, 4289.000000, -10000.000000, 4303.000000, 3995.000000, -10000.000000, 3847.000000, -10000.000000, 4242.000000, -10000.000000, -10000.000000}};

void ker1(int k, int f, float* X){
    for (uint i = 0; i < k; i++){
        for (uint j = 0; j < N; j++){
            float ind = mappingindices[j];
            if(i==0){
                X[i*N+j] = 1.0;
            } else if(i==1){
                X[i*N+j] = ind;
            } else {
                float ip = (float)(i / 2);
                float jp = ind;
                float angle = 2 * PI * ip * jp / f;
                if(i%2 == 0) {
                    X[i*N+j] = sin(angle);
                } else {
                    X[i*N+j] = cos(angle);
                }
            }
        }
    }
}

int main(int argc, char const *argv[]) {

	// if (argc != 8) {
	// 	printf("Not enough arguments!\n");
    //   	return -1;
	// }

	// int trend 			= atoi(argv[1]);
	// int k 				= atoi(argv[2]);
	// int m 				= atoi(argv[3]);
	// int n 				= atoi(argv[4]);
	// int N 				= atoi(argv[5]);
	// float freq 			= atoi(argv[6]);
	// float hfrac 		= atoi(argv[7]);
	// float lam 			= atoi(argv[8]);

	printf("%d\n", trend);
	printf("%d\n", k);
	printf("%d\n", m);
	printf("%d\n", n);
	printf("%d\n", N);
	printf("%f\n", freq);
	printf("%f\n", hfrac);
	printf("%f\n", lam);
	printf("%lu\n", sizeof(mappingindices)/sizeof(mappingindices[0]));
	printf("%lu\n", sizeof(sample)/sizeof(sample[0]));

    int k2p2 = 2*k +2;
    float X[k2p2][N];
    ker1(k2p2,N,&X);

    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            printf("%f, ", X[i*N+j]);
        }
    }
    printf("\n");



	return 0;
}



