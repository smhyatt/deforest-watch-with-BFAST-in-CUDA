#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

#define PI 3.14159265
#define F32_MIN -FLT_MAX
#define I32_MIN -2147483648
typedef unsigned int uint;

// (trend: i32) (k: i32) (n: i32) (freq: f32) (hfrac: f32) (lam: f32) (mappingindices : [N]i32) (images : [m][N]f32)
// (trend, k, m, n, N, freq, hfrac, lam, mappingindices, sample)

// peru
// int trend   = 1;
// int k 	   = 3;
// int m 	   = 2;
// int n 	   = 113;
// int N 	   = 235;
// float freq  = 365.000000;
// float hfrac = 0.250000;
// float lam   = 1.736126;

// sahara
int trend   = 1;
int k       = 3;
int m       = 2;
int n       = 228;
int N       = 414;
float freq  = 12.000000;
float hfrac = 0.250000;
float lam   = 1.736126;


// peru
// int mappingindices[235] = {122, 138, 170, 202, 218, 250, 266, 314, 330, 378, 394, 474, 490, 570, 586, 602, 618, 666, 730, 1290, 1306, 1322, 1338, 1354, 1370, 1386, 1402, 1482, 1594, 1642, 1674, 1706, 1722, 1802, 1818, 1850, 1898, 1994, 2010, 2042, 2074, 2146, 2154, 2194, 2234, 2274, 2322, 2330, 2338, 2346, 2378, 2410, 2418, 2426, 2442, 2450, 2458, 2474, 2490, 2586, 2594, 2650, 2658, 2682, 2690, 2698, 2706, 2722, 2730, 2738, 2754, 2762, 2770, 2778, 2794, 2810, 2818, 2826, 2834, 2898, 2930, 3034, 3050, 3058, 3066, 3074, 3082, 3106, 3122, 3130, 3146, 3154, 3178, 3194, 3242, 3266, 3274, 3354, 3370, 3418, 3426, 3458, 3466, 3498, 3514, 3522, 3530, 3538, 3562, 3602, 3610, 3626, 3650, 3778, 3786, 3810, 3818, 3874, 3882, 3890, 3906, 3922, 3938, 3946, 3962, 3970, 3994, 4002, 4090, 4138, 4210, 4218, 4242, 4250, 4258, 4266, 4274, 4282, 4290, 4306, 4354, 4434, 4466, 4482, 4514, 4530, 4546, 4562, 4578, 4594, 4610, 4626, 4642, 4658, 4722, 4738, 4858, 4866, 4874, 4890, 4922, 4994, 5002, 5018, 5026, 5042, 5050, 5058, 5082, 5106, 5114, 5130, 5138, 5162, 5202, 5210, 5234, 5250, 5282, 5290, 5298, 5306, 5322, 5362, 5378, 5386, 5410, 5418, 5426, 5442, 5450, 5458, 5466, 5514, 5530, 5562, 5618, 5626, 5642, 5650, 5658, 5666, 5682, 5690, 5698, 5706, 5714, 5730, 5738, 5746, 5754, 5762, 5786, 5802, 5810, 5818, 5858, 5866, 5874, 5914, 5946, 5970, 5978, 5994, 6010, 6026, 6034, 6042, 6050, 6066, 6074, 6082, 6090, 6098, 6114};
// sahara
int mappingindices[414] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414};
// peru
// float sample[2][235] = {{F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 4716.000000, F32_MIN, 4147.000000, 4546.000000, F32_MIN, 3565.000000, F32_MIN, F32_MIN, F32_MIN, 4208.000000, 4446.000000, F32_MIN, 3653.000000, 3734.000000, 4064.000000, F32_MIN, F32_MIN, 4628.000000, 4034.000000, F32_MIN, 3867.000000, 2117.000000, 4598.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 4210.000000, F32_MIN, 5253.000000, 4219.000000, F32_MIN, 3074.000000, F32_MIN, F32_MIN, 3982.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 3888.000000, F32_MIN, F32_MIN, F32_MIN, 3662.000000, 4182.000000, 4475.000000, 4659.000000, 5102.000000, 4552.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 3827.000000, F32_MIN, F32_MIN, F32_MIN, 3773.000000, 3985.000000, F32_MIN, F32_MIN, 5331.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 4165.000000, F32_MIN, 3965.000000, F32_MIN, 4094.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 3778.000000, F32_MIN, 3176.000000, 4073.000000, 4461.000000, 4017.000000, 4031.000000, 3929.000000, 3999.000000, F32_MIN, 4349.000000, 4182.000000, F32_MIN, F32_MIN, F32_MIN, 3001.000000, 3286.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 4212.000000, 4309.000000, F32_MIN, 4451.000000, F32_MIN, F32_MIN, 3999.000000, 4064.000000, F32_MIN, F32_MIN, 4215.000000, F32_MIN, F32_MIN, F32_MIN, 3917.000000, 4035.000000, 4263.000000, 4286.000000, F32_MIN, F32_MIN, 4062.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 4129.000000, 4042.000000, F32_MIN, 3923.000000, F32_MIN, 4407.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 4716.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 4690.000000, 4570.000000, 4505.000000, F32_MIN, 4172.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 4225.000000, F32_MIN, F32_MIN, F32_MIN, 4402.000000, 4168.000000, 4053.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 4237.000000, F32_MIN, F32_MIN, F32_MIN, 4035.000000, 3533.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 4050.000000, 4341.000000, F32_MIN, 4555.000000, 4096.000000, F32_MIN, 4077.000000, 4196.000000, F32_MIN, F32_MIN, 4205.000000, 4647.000000, F32_MIN, 4005.000000, 3607.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 3977.000000, 3761.000000, F32_MIN, F32_MIN, F32_MIN, 4184.000000, 2652.000000, 4341.000000, F32_MIN, F32_MIN, 4476.000000, 4257.000000, F32_MIN, 4066.000000, F32_MIN, F32_MIN, F32_MIN, 4836.000000, F32_MIN, F32_MIN, 4048.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 4533.000000, 4330.000000, F32_MIN, 4554.000000, 4100.000000, F32_MIN, 3692.000000, F32_MIN, 4337.000000, F32_MIN, F32_MIN}, {F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 4581.000000, F32_MIN, 4318.000000, 4602.000000, F32_MIN, 3480.000000, F32_MIN, F32_MIN, F32_MIN, 4236.000000, 3872.000000, F32_MIN, 4249.000000, 4039.000000, 3906.000000, F32_MIN, F32_MIN, 4074.000000, 3956.000000, F32_MIN, 4107.000000, 1820.000000, 3989.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 4152.000000, F32_MIN, 4480.000000, 3969.000000, F32_MIN, 3703.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 3888.000000, F32_MIN, F32_MIN, F32_MIN, 3728.000000, 3905.000000, 4534.000000, 4337.000000, 4563.000000, 4268.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 3717.000000, F32_MIN, F32_MIN, F32_MIN, 3557.000000, 3607.000000, F32_MIN, F32_MIN, 4753.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 4432.000000, F32_MIN, 4251.000000, F32_MIN, 4159.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 3535.000000, F32_MIN, 3275.000000, 4024.000000, 4886.000000, 4409.000000, 4170.000000, 4220.000000, 4211.000000, F32_MIN, 4320.000000, 4102.000000, F32_MIN, F32_MIN, F32_MIN, 3058.000000, 3678.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 4426.000000, 4250.000000, F32_MIN, 4244.000000, F32_MIN, F32_MIN, 3995.000000, 3987.000000, F32_MIN, F32_MIN, 4078.000000, F32_MIN, F32_MIN, F32_MIN, 4093.000000, 4165.000000, 4322.000000, 4239.000000, F32_MIN, F32_MIN, 3718.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 4234.000000, 3894.000000, F32_MIN, 3684.000000, F32_MIN, 4341.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 4805.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 4369.000000, 4538.000000, 4351.000000, F32_MIN, 4270.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 4115.000000, F32_MIN, F32_MIN, F32_MIN, 4373.000000, 4262.000000, 4005.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 4261.000000, F32_MIN, F32_MIN, F32_MIN, 4027.000000, 3533.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 3866.000000, 4232.000000, F32_MIN, 4513.000000, 4038.000000, F32_MIN, 4164.000000, 4045.000000, 3489.000000, F32_MIN, 3896.000000, 4625.000000, F32_MIN, 4224.000000, 3571.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 3719.000000, 3361.000000, F32_MIN, F32_MIN, F32_MIN, 4295.000000, 2625.000000, 4151.000000, F32_MIN, F32_MIN, 4310.000000, 4059.000000, F32_MIN, 4152.000000, F32_MIN, F32_MIN, F32_MIN, 4584.000000, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 4334.000000, 4289.000000, F32_MIN, 4303.000000, 3995.000000, F32_MIN, 3847.000000, F32_MIN, 4242.000000, F32_MIN, F32_MIN}};
// sahara
float sample[2*414] = {F32_MIN, F32_MIN, 6939.676758, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 7315.675781, 6989.361328, 7455.587402, F32_MIN, 7779.796875, F32_MIN, 7474.850586, 6530.423828, 6101.163086, 5259.306641, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 6078.483398, F32_MIN, F32_MIN, F32_MIN, 5411.000977, 6439.008301, 5363.949219, F32_MIN, F32_MIN, 4539.385254, F32_MIN, F32_MIN, 5415.133789, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 7585.372070, 5499.139648, 4980.914551, 5724.080566, 7087.082520, F32_MIN, F32_MIN, 4368.201172, F32_MIN, F32_MIN, F32_MIN, 7744.390625, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 5796.133301, 5148.380371, 5470.048340, F32_MIN, 6894.774414, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 6646.684570, F32_MIN, F32_MIN, 4783.465332, 6657.979980, F32_MIN, 7733.155273, F32_MIN, 5355.178711, 7835.649902, 6655.700195, 5304.320801, 4864.532227, F32_MIN, 4489.860352, F32_MIN, 6032.019043, 7596.905273, 6228.702637, F32_MIN, 4804.752930, F32_MIN, F32_MIN, 4691.677246, 7948.397461, 5084.504883, 6130.926758, 5980.913086, 4641.091797, 6147.282227, 7455.835938, 4669.083008, 5297.273926, F32_MIN, 5408.963867, 4099.746582, F32_MIN, F32_MIN, F32_MIN, 7919.689453, F32_MIN, 7847.644531, F32_MIN, 4476.773438, 6334.089844, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 6438.109375, F32_MIN, 7150.950195, 7522.899902, F32_MIN, 6349.108398, F32_MIN, 4818.277344, F32_MIN, F32_MIN, F32_MIN, 6046.610352, F32_MIN, 5421.979492, F32_MIN, 6959.597168, F32_MIN, 4022.170898, F32_MIN, 5652.712402, F32_MIN, 6355.249023, F32_MIN, F32_MIN, 6172.537109, 6529.701660, 7236.536133, 6829.612305, 4199.769043, F32_MIN, F32_MIN, 4502.702148, F32_MIN, 4018.161377, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 4011.584229, F32_MIN, F32_MIN, 7646.160156, 7809.455078, 6224.596680, F32_MIN, F32_MIN, F32_MIN, 5710.611328, F32_MIN, 4831.783203, F32_MIN, 7571.867188, F32_MIN, F32_MIN, F32_MIN, 5437.691406, 6802.190430, F32_MIN, 7473.796875, 7663.843750, 5411.232910, 5624.748535, F32_MIN, F32_MIN, F32_MIN, 4654.756836, 5777.149414, F32_MIN, F32_MIN, F32_MIN, 4021.627197, 7963.641113, F32_MIN, F32_MIN, 6460.788086, F32_MIN, F32_MIN, 5821.801758, 4193.262695, F32_MIN, 5174.064941, 5286.193359, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 5860.827148, 7973.592773, 5289.646484, F32_MIN, 7260.789062, 5542.467773, F32_MIN, 5456.207520, 4599.462891, 4661.252930, F32_MIN, F32_MIN, 4779.471680, 5880.367676, F32_MIN, F32_MIN, 5136.988281, F32_MIN, F32_MIN, 5591.050781, F32_MIN, F32_MIN, 4885.754395, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 7617.345703, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 6730.458008, 5953.076172, F32_MIN, 6053.517578, F32_MIN, F32_MIN, 6023.134277, 6716.538086, 6006.648438, 6921.358398, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 7422.293945, 5561.504395, 7370.412109, 5154.032715, 6304.627930, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 7063.054688, 4731.649414, 5441.925781, 7197.659180, 6196.366211, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 6889.468750, F32_MIN, F32_MIN, 6232.260742, F32_MIN, 5092.247559, 7878.941895, F32_MIN, 5784.416016, 7533.649414, F32_MIN, 5855.939453, 4055.066650, 6123.016602, 4146.056152, F32_MIN, 5673.895020, 4589.479492, 6757.052734, 5701.538086, 6938.203125, 5992.674805, F32_MIN, F32_MIN, 4904.734375, 4436.969727, F32_MIN, F32_MIN, F32_MIN, 6217.124023, F32_MIN, F32_MIN, F32_MIN, 7625.854492, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 6735.142578, 4047.844727, 5513.336914, F32_MIN, 4147.819336, 7377.392090, F32_MIN, F32_MIN, 6502.716797, F32_MIN, 6553.272949, F32_MIN, F32_MIN, 4189.100098, F32_MIN, F32_MIN, F32_MIN, 4341.377441, 6624.422852, F32_MIN, 4837.254883, 7137.864746, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 4982.037109, F32_MIN, 6797.025391, F32_MIN, 6397.886719, F32_MIN, F32_MIN, F32_MIN, 4786.889648, 7941.480469, 7195.894531, F32_MIN, F32_MIN, 5969.043945, 7709.972656, 4076.921631, F32_MIN, F32_MIN, 5561.660156, 6890.989746, 6970.568359, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 7938.936523, F32_MIN, 5217.643555, F32_MIN, 7012.348633, 5094.632812, 7008.682129, F32_MIN, F32_MIN, 4352.267090, F32_MIN, 4107.474609, 7906.010742, 7023.367188, 4963.508301, 5502.608887, 4442.437012, 4867.733398, 6349.615723, F32_MIN, F32_MIN, 7673.832031, F32_MIN, 5024.297852, F32_MIN, 7695.417480, 5499.181641, 6983.608398, 5775.124023, F32_MIN, 7245.734375, F32_MIN, 6171.529297, F32_MIN, F32_MIN, 113.816185, 4021.152344, 44.238491, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 262.869141, F32_MIN, F32_MIN, 942.577026, 4136.626465, 4096.059082, F32_MIN, F32_MIN, F32_MIN, 7174.472168, F32_MIN, 4399.747559, F32_MIN, 4075.672363, 4775.550781, F32_MIN, F32_MIN, 4172.123535, 4566.777344, F32_MIN, 6025.019531, F32_MIN, 7304.147461, F32_MIN, F32_MIN, 4441.420410, 7812.694824, F32_MIN, 4912.644531, F32_MIN, 5912.727539, F32_MIN, 7871.999512, 5291.206543, 7836.223145, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 4389.689453, F32_MIN, F32_MIN, 5555.194824, 6805.528320, F32_MIN, 7236.161133, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 7234.142578, F32_MIN, 6352.510742, 6026.826660, F32_MIN, 7420.816895, 6247.645020, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 7724.103516, 6192.547363, F32_MIN, F32_MIN, F32_MIN, 7138.182617, 7209.364258, 7202.844727, 4525.137695, F32_MIN, F32_MIN, F32_MIN, 4987.536621, F32_MIN, F32_MIN, 7790.692383, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 6747.841797, 5064.946289, F32_MIN, 5779.428223, F32_MIN, F32_MIN, 6456.619141, 7445.861816, 7191.064453, 7866.624023, 5820.928711, 6050.909180, 7431.282715, F32_MIN, 4985.047852, F32_MIN, F32_MIN, 4874.580566, 4869.142578, 6394.104492, F32_MIN, 7422.007812, F32_MIN, 7680.481445, F32_MIN, 5124.959473, 6913.575195, 5193.621582, 5314.999023, F32_MIN, 6586.767578, F32_MIN, 4379.528809, 4243.517578, F32_MIN, 5342.512695, 4445.591309, 5126.887695, 4007.475830, F32_MIN, F32_MIN, 4043.859375, 5134.572754, 6967.506836, 4531.345215, F32_MIN, 6490.901855, F32_MIN, 6164.566406, F32_MIN, F32_MIN, F32_MIN, 7053.101562, 4270.981934, F32_MIN, F32_MIN, 6028.870117, 7579.613281, F32_MIN, F32_MIN, 6400.463867, 4775.544434, 4308.828125, F32_MIN, F32_MIN, F32_MIN, 4077.097412, F32_MIN, 5086.316406, 5594.733887, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 7540.441406, 4640.498535, F32_MIN, 6769.741211, 6191.127441, 7914.182617, F32_MIN, 7470.567871, 7781.246094, 4522.742188, F32_MIN, 6476.835938, 7352.247070, 4307.372070, 5148.600586, F32_MIN, 7211.527344, 7627.603027, F32_MIN, 4958.299316, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 5580.769531, F32_MIN, F32_MIN, 7682.332031, 5835.822266, 4988.720703, 6533.589844, F32_MIN, F32_MIN, F32_MIN, 5166.587891, 4368.135254, F32_MIN, 4778.992188, 6738.792969, 4280.257812, F32_MIN, F32_MIN, 7601.021484, F32_MIN, 6891.837891, 7896.775391, 5253.977539, 6761.024902, 5432.757324, 4637.040039, F32_MIN, 5210.212891, 6169.062988, 6842.950684, 4067.747803, F32_MIN, F32_MIN, 7142.861816, 5081.511230, F32_MIN, 5198.092773, 5128.000488, 5710.951172, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 6779.713379, 7539.762695, 7898.147949, F32_MIN, 5291.704590, F32_MIN, 5486.365723, 4363.476562, 5389.142578, F32_MIN, 5777.224609, F32_MIN, F32_MIN, 7412.049805, F32_MIN, F32_MIN, 6022.679688, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 7130.449707, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 4544.791016, F32_MIN, F32_MIN, F32_MIN, 7215.165039, 7252.212891, 7558.048828, F32_MIN, 6412.936523, 6842.205078, F32_MIN, F32_MIN, F32_MIN, 6653.300293, F32_MIN, F32_MIN, 3784.883545, F32_MIN, 3147.578857, F32_MIN, 3314.653320, 1633.028687, 2925.905762, 1404.803955, F32_MIN, F32_MIN, 3611.196533, 1071.793091, F32_MIN, 4698.238770, F32_MIN, 4430.290039, 4531.154785, 3372.349121, F32_MIN, 368.608002, 3077.385742, F32_MIN, 1818.106812, F32_MIN, F32_MIN, 1190.725586, F32_MIN, 4231.801270, 3281.938477, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 3452.274902, F32_MIN, 143.528214, F32_MIN, F32_MIN, 1106.386597, F32_MIN, 2724.151123, F32_MIN, 3642.510742, 2643.082764, F32_MIN, 2095.221924, F32_MIN, 4181.152344, 3405.392090, 1680.265869, 3112.684082, F32_MIN, F32_MIN, 343.798431, F32_MIN, F32_MIN, F32_MIN, 3914.347656, F32_MIN, 3869.361084, F32_MIN, F32_MIN, F32_MIN, 804.652100, F32_MIN, F32_MIN, 3091.945312, 1296.245850, F32_MIN, 2496.298584, 3832.327393, F32_MIN, F32_MIN, 3732.827881, 2331.603760, 3842.329346, 3079.394043, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 2098.337158, F32_MIN, 1160.595215, 3091.035156, 2362.670410, 3462.324951, 4896.783203, 2625.785156, F32_MIN, F32_MIN, 155.583786, 184.791733, F32_MIN, F32_MIN, 595.746460, 2279.479004, F32_MIN, F32_MIN, 2571.462158, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 4709.876953, 469.709106, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, F32_MIN, 3404.765625, 1440.361450, 2684.044434, F32_MIN, F32_MIN, 694.205078, 4972.682617, 1361.105835, 1942.933838, 2358.591797, F32_MIN, F32_MIN, 2309.950928, 3640.333740, F32_MIN, F32_MIN, 2708.921631, F32_MIN, 2392.310547, F32_MIN, F32_MIN, 2046.303101, F32_MIN, 200.289688, 3183.757568, 3170.747803, F32_MIN, 2933.344482, F32_MIN, 1191.964844};

void ker1(int kp, int f, float* X){
    for (uint i = 0; i < kp; i++){
        for (uint j = 0; j < N; j++){
            float ind = mappingindices[j];
            uint index = i*N + j;
            if(i==0){
                X[index] = 1.0;
            } else if(i==1){
                X[index] = ind;
            } else {
                float ip = (float)(i / 2);
                float jp = ind;
                float angle = 2 * PI * ip * jp / f;
                if(i%2 == 0) {
                    X[index] = sin(angle);
                } else {
                    X[index] = cos(angle);
                }
            }
        }
    }
}


void transpose(int K, float* X, float* XT) {
    for (uint i = 0; i < K; i++){
        for (uint j = 0; j < N; j++){
            uint Xidx  = i*N + j;
            uint XTidx = j*K + i;
            XT[XTidx]  = X[Xidx];
        }
    }
}


// let dotprod_filt [n] (vct: [n]f32) (xs: [n]f32) (ys: [n]f32) : f32 =
//   f32.sum (map3 (\v x y -> x * y * if (f32.isnan v) then 0.0 else 1.0) vct xs ys)
float dotProdFilt(float* Xvct, float* XTvct, float* yvct) {
    float acc = 0.0;
    for (uint i = 0; i < n; i++) {
        if (yvct[i] != F32_MIN) {
            acc += Xvct[i] * XTvct[i];
        }
    }
    return acc;
}



// let matmul_filt [n][p][m] (xss: [n][p]f32) (yss: [p][m]f32) (vct: [p]f32) : [n][m]f32 =
//   map (\xs -> map (dotprod_filt vct xs) (transpose yss)) xss
// [p][m]
// [3][2]
// XT = [[1,2],
//       [3,4],
//       [5,6]]
// [n][p]
// [2][3]
// X  = [[1,3,5],
//       [2,4,6]]
// y  = [8,9,7]
// xss = Xn, yss = XTn, vct = y
void mmMulFilt(float* X, float* XT, float* y, float* Xsqr, uint K){
    float* tspVct = calloc(n,sizeof(float));

    // K
    for (int i = 0; i < K; i++) {
        // K
        for (int j = 0; j < K; j++) {
            uint XIdx = i*N;
            uint XTIdx = j;
            uint resIdx = i*K + j;

            for (uint l = 0; l < n; l++) {
                uint idx  = l*K + j;
                tspVct[l] = XT[idx];
            }

            Xsqr[resIdx] = dotProdFilt(&X[XIdx], tspVct, y);
        }
    }

    // free(tspVct);
}


// -- Xsqr,Xsqr−1:[K][K]f32; β0,β:[K]f32
// let Xsqr = mmMulFilt X[:,:n] XT[:n,:] y[:n] -- ker 2
void ker2(float* X, float* XT, float* Xsqr, uint K) {
    for (uint pix = 0; pix < m; pix++) {
        mmMulFilt(X, XT, &sample[pix*N], &Xsqr[pix*K*K], K);
    }

}


// ----------
//   let Xinv = intrinsics.opaque <|
//              map mat_inv Xsqr

//   let gauss_jordan [nm] (n:i32) (m:i32) (A: *[nm]f32): [nm]f32 =
//     loop A for i < n do
//       let v1 = A[i]
//       let A' = map (\ind -> let (k, j) = (ind / m, ind % m)
//                             in if v1 == 0.0 then unsafe A[k*m+j] else
//                             let x = unsafe (A[j] / v1) in
//                                 if k < n-1  -- Ap case
//                                 then unsafe ( A[(k+1)*m+j] - A[(k+1)*m+i] * x )
//                                 else x      -- irow case
//                    ) (iota nm)
//       in  scatter A (iota nm) A'


// procedure Naive Gauss(n, (ai j ), (bi ), (xi ))
// integer i, j, k, n; real sum, xmult
// real array (ai j )1:n×1:n , (bi )1:n , (xi )1:n
// for k = 1 to n − 1 do           // row
//     for i = k + 1 to n do       // row we are working on
//         xmult ← aik/akk
//         aik ← xmult
//         for j = k + 1 to n do   // column
//             aij ← aij −(xmult)akj
//         end for
//         bi ← bi − (xmult)bk
//     end for
// end for
// xn ←bn/ann
// for i =n−1to1 step−1do
//     sum ← bi
//     for j = i + 1 to n do
//         sum ← sum − ai j x j
//     end for
// xi ← sum/aii
// end for
// end procedure Naive Gauss

void gaussJordan(float* XsqrInv, uint cols, uint K){
    // Making the upper triangle
    for (uint row = 0; row < K-1; row++){
        for (uint rowWork = row + 1; rowWork < K; rowWork++){
            float xMult = XsqrInv[rowWork*cols+row] / XsqrInv[row*cols+row];
            for (uint col = row; col < cols; col++){
                uint factorIdx   = row     * cols + col;
                uint elemIdx     = rowWork * cols + col;
                XsqrInv[elemIdx] = XsqrInv[elemIdx] - xMult * XsqrInv[factorIdx];
            }
        }
    }

    // scalling
    for (uint i = 0; i < K; i++) {
        float temp = XsqrInv[i*cols+i];
        for (uint j = i; j < cols; j++){
            XsqrInv[i*cols+j] = XsqrInv[i*cols+j] / temp;
        }
    }

    // Making back substitution
    for (uint row = K-1; row >= 1; row--){
        for (int rowWork = row-1; 0 <= rowWork ; rowWork--){
            float x = XsqrInv[rowWork*cols+row] / XsqrInv[row*cols+row];
            for (uint col = row; col < cols; col++){
                XsqrInv[rowWork*cols+col] = XsqrInv[rowWork*cols+col] - XsqrInv[row*cols+col] * x;
            }
        }
    }
}


void doubleDown(float* XsqrInv, float* XsqrInvLess, uint K) {
    for (int pix = 0; pix < m; pix++) {
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                uint XinvIdx  = pix*K*(2*K) + i*(K*2) + j+K; // XsqrInv er K længere
                uint XlessIdx = pix*K*K + i*K + j;
                XsqrInvLess[XlessIdx] = XsqrInv[XinvIdx];
            }
        }
    }
}


void mkXsqr(float* Xsqr, float* XsqrInv, uint K) {
    uint cols = 2*K;        // 2*8=16
    uint identIdx = K*cols; // 8*16=128

    for (uint i = 0; i < K; i++){
        for (uint j = 0; j < K; j++){
            // 1*8+0
            uint sqrIdx = i*K + j;
            // 1*16
            uint invIdx = i*cols + j;
            XsqrInv[invIdx] = Xsqr[sqrIdx];
        }
    }

    for (uint i = 0; i < K; i++){
        for (uint j = K; j < identIdx; j+=cols+1){
            uint idx = i*identIdx + j;
            XsqrInv[idx] = 1.0;
        }
    }
}


//   let mat_inv [n] (A: [n][n]f32): [n][n]f32 =
//     let m = 2*n
//     let nm= n*m
//     -- Pad the matrix with the identity matrix.
//     let Ap = map (\ind -> let (i, j) = (ind / m, ind % m)
//                           in  if j < n then unsafe ( A[i,j] )
//                                        else if j == n+i
//                                             then 1.0
//                                             else 0.0
//                  ) (iota nm)
//     let Ap' = gauss_jordan n m Ap
//     -- Drop the identity matrix at the front!
//     in (unflatten n m Ap')[0:n,n:2*n]

void ker3(float* Xsqr, float* XsqrInv, uint K){
    uint cols = 2*K;        // 2*8=16
    uint identIdx = K*cols; // 8*16=128

    for (int pix = 0; pix < m; pix++) {
        mkXsqr(&Xsqr[pix*(K*K)], &XsqrInv[pix*identIdx], K);
        gaussJordan(&XsqrInv[pix*identIdx], cols, K);
    }

    // for (uint ind = 0; ind < identIdx; ind++){
    //     // (i, j) = (ind / m, ind % m)
    //     uint iIdx = ind / cols;
    //     uint jIdx = ind % cols;
    //     uint invIdx = iIdx*identIdx*sizeof(float) + jIdx*sizeof(float);
    //     uint sqrIdx = iIdx*K*sizeof(float) + jIdx*sizeof(float);
    //     if(jIdx<K){
    //         XsqrInv[invIdx] = Xsqr[sqrIdx];
    //     } else {
    //         if(jIdx==K+iIdx){
    //             XsqrInv[invIdx] = 1.0;
    //         } else {
    //             XsqrInv[invIdx] = 0.0;
    //         }
    //     }
    // }

}



// let matvecmul_row_filt [n][m] (xss: [n][m]f32) (ys: [m]f32) =
//     map (\xs -> map2 (\x y -> if (f32.isnan y) then 0 else x*y) xs ys |> f32.sum) xss

void mvMulFilt(float* X, float* y, uint K, float* B0){

    for (int i = 0; i < K; i++) {
        uint XIdx  = i*N;

        B0[i] = dotProdFilt(&X[XIdx], y, y);
    }
}



// let beta0  = map (matvecmul_row_filt Xh) Yh   -- [2k+2]
//                |> intrinsics.opaque
// let β0 = mvMulFilt X[:,:n] y[:n]
void ker4(float* X, uint K, float* B0){
    for (uint pix = 0; pix < m; pix++) {
        mvMulFilt(X, &sample[pix*N], K, &B0[pix*K]);
    }
}



void mvMul(float* M, float* v, uint rows, uint cols, float* res_v) {

    // for ker6 er rows = N og cols = K
    for (int i = 0; i < rows; i++) {
        float acc = 0.0;

        for (uint elm = 0; elm < cols; elm++) {
            uint MIdx = i*cols + elm;
            acc += M[MIdx] * v[elm];
        }
        res_v[i] = acc;
    }
}


// let β = mvMul Xsqr−1 β0
void ker5(float* XsqrInvLess, uint K, float* B0, float* B){
    for (uint pix = 0; pix < m; pix++) {
        mvMul(&XsqrInvLess[pix*(K*K)], &B0[pix*K], K, K, &B[pix*K]);
    }
}


// -- yˆ,r,I : [N]f32
// let yˆ = mvMul XT β
void ker6(float* XT, float* B, uint K, float* yhat) {
    for (uint pix = 0; pix < m; pix++) {
        mvMul(XT, &B[pix*K], N, K, &yhat[pix*N]);
    }

}



// -- filterPadWithKeys (\y -> !(f32.isnan y)) (f32.nan) y_error_all
// -- Input:   p:(p->value:true or nan:false) dummy:nan arr:[nan,float,nan,float]
// -- Returns: ([(float,int),(float,int)],int)
// let filterPadWithKeys [n] 't
//            (p : (t -> bool))
//            (dummy : t)
//            (arr : [n]t) : ([n](t,i32), i32) =
//   -- [0,1,0,1] <- [nan,float,nan,float]
//   let tfs = map (\a -> if p a then 1 else 0) arr
//   -- number of valid
//   let isT = scan (+) 0 tfs
//   let i   = last isT
//   -- isT:  [0,1,1,2]
//   -- inds: [-1,0,-1,1]
//   let inds= map2 (\a iT -> if p a then iT-1 else -1) arr isT
//   --X [nan,nan,nan,nan]
//   --I inds: [-1,0,-1,1]
//   --D [nan,float,nan,float]
//   --R [float,float,nan,nan]
//   let rs  = scatter (replicate n dummy) inds arr
//   --X [0,0,0,0]
//   --I inds: [-1,0,-1,1]
//   --D [0,1,2,3]
//   --R [1,3,0,0]
//   let ks  = scatter (replicate n 0) inds (iota n)
//   in  (zip rs ks, i)


void filterNaNsWKeys(float* diffVct, uint* valid, float* y_errors, int* val_indss) {
    uint idx = 0;
    *valid   = 0;

    for (int i = 0; i < N; i++) {
        uint check = (diffVct[i] != F32_MIN);
        *valid    += check;
        uint ind   = (check * (*valid) - 1);

        if (ind != -1) {
            y_errors[idx]  = diffVct[i];
            val_indss[idx] = i;
            idx++;
        }
    }

    // for (int i = 0; i < N; i++) {
    //     tfs[i]  = (diffVct[i] != NAN);
    //     acc    += tfs[i];
    //     inds[i] = tfs[i] * acc - 1;
    //     if (inds[i] != -1) {
    //         y_errors[idx]    = diffVct[i];
    //         val_indss[idx] = i;
    //         idx++;
    //     }
    // }

}

  // let (Nss, y_y_errors, val_indss) = ( intrinsics.opaque <| unzip3 <|
  //   -- y p
  //   map2 (\y y_pred ->
  //           let y_error_all = zip y y_pred |>
  //               map (\(ye,yep) -> if !(f32.isnan ye)
  //                                 then ye-yep else f32.nan )
  //           -- [nan,dif,nan,dif]
  //           let (tups, Ns) = filterPadWithKeys (\y -> !(f32.isnan y)) (f32.nan) y_error_all
  //           -- (tups:([false,],[nan,dif,nan,dif]), Ns:float#ofvalid)
  //           let (y_error, val_inds) = unzip tups
  //           in  (Ns, y_error, val_inds)
  //        ) images y_preds )


// let (N,r,I)= map2 (-) y yˆ |> filterNaNsWKeys
void ker7(float* yhat, float* y_errors_all, uint* Nss, float* y_errors, int* val_indss) {
    for (uint pix = 0; pix < m; pix++) {
        for (uint i = 0; i < N; i++) {
            float y  = sample[pix*N+i];
            float yh = yhat[pix*N + i];

            if (y != F32_MIN) {
                y_errors_all[pix*N + i] = y-yh;
            } else {
                y_errors_all[pix*N + i] = F32_MIN;
            }
        }

        filterNaNsWKeys(&y_errors_all[pix*N], &Nss[pix], &y_errors[pix*N], &val_indss[pix*N]);

    }
}



void comp(float* yh, float* y_errors, uint K, int* hs, uint* nss, float* sigmas) {
    float acc = 0.0;

    for (uint i = 0; i < n; i++) {
        *nss += (yh[i] != F32_MIN);
    }

    for (uint j = 0; j < n; j++) {
        if (j < *nss) {
            float y_err = y_errors[j];
            acc += y_err*y_err;
        }
    }
    *sigmas = sqrt(acc/((float)(*nss-K)));
    *hs = (int)(((float)*nss) * hfrac);
}



void ker8(float* y_errors, uint K, int* hs, uint* nss, float* sigmas) {
    for (uint pix = 0; pix < m; pix++) {
        comp(&sample[pix*N], &y_errors[pix*N], K, &hs[pix], &nss[pix], &sigmas[pix]);
    }
}


void MO_fsts_comp(int hmax, int* hs, float* y_errors, uint* nss, float* MO_fsts) {

    for (int i = 0; i < hmax; i++) {

        if (i < *hs) {
            uint idx = i + *nss - *hs + 1;
            *MO_fsts += y_errors[idx];
        }
    }
}


void ker9(int* hs, float* y_errors, uint* nss, float* MO_fsts) {
    int hmax = I32_MIN;
    for (int i = 0; i < m; i++) {
        int cur = hs[i];
        if (cur >= hmax) {
            hmax = cur;
        }
    }

    for (uint pix = 0; pix < m; pix++) {
        MO_fsts_comp(hmax, &hs[pix], &y_errors[pix*N], &nss[pix], &MO_fsts[pix]);
    }
}

float logplus(float x){
    if(x>exp(1.0)){
        return log(x);
    } else {
        return 1.0;
    }
}


void compBound(float* b){
    for (uint i = 0; i < (N-n); i++){
        uint t = n+i;
        int time = mappingindices[t];
        float tmp = logplus((float)time / (float)mappingindices[N-1]);
        b[i] = lam * sqrt(tmp);
    }
}

void MO_comp(uint Nmn, int* h, float* MO_fst, float* y_error, uint* Ns,
             uint* ns, float* MO){
    float acc = 0.0;
    for (uint i = 0; i < Nmn; i++){
        if(i >= *Ns-*ns){
            MO[i] = acc;
        } else if(i==0) {
            acc += *MO_fst;
            MO[i] = acc;
        } else {
            acc += - y_error[*ns - *h + i] + y_error[*ns + i];
            MO[i] = acc;
        }
    }
}


void MO_prime_comp(float* MO, uint* ns, float sigma, float* MOp){
    for (uint i = 0; i < N-n; i++){
        float mo = MO[i];
        MOp[i] = mo / (sigma * (sqrt( (float)*ns )));
    }
}


//           let fst_break' = if !is_break then -1
//                              else let adj_break = adjustValInds n ns Ns val_inds fst_break
//                                   in  ((adj_break-1) / 2) * 2 + 1  -- Cosmin's validation hack
//             let fst_break' = if ns <=5 || Ns-ns <= 5 then -2 else fst_break'
void breaks(float* MOp, float* bound, uint Ns, uint ns, uint Nmn, int* isBreak, int* fstBreak){
    int i = 0;

    for (uint i = 0; i < Nmn; i++){
        float mop = MOp[i];

        if(i < (Ns-ns) && mop != F32_MIN){
            if (fabsf(mop) > bound[i] == 1) {
                *isBreak  = 1;
                *fstBreak = i;
                break;
            }
        }
    }
}


void meanComp(uint Ns, uint ns, uint Nmn, float* MOp, float* mean) {
    for (uint i = 0; i < Nmn; i++) {
        if (i < (Ns-ns)) {
            *mean += MOp[i];
        }
    }
}


// let adjustValInds [N] (n : i32) (ns : i32) (Ns : i32) (val_inds : [N]i32) (ind: i32) : i32 =
//     if ind < Ns - ns then (unsafe val_inds[ind+ns]) - n else -1
int adjustValInds(uint ns, uint Ns, int* val_inds, int fstBreak) {
    if (fstBreak < Ns-ns) {
        return (val_inds[fstBreak+ns]-n);
    } else {
        return -1;
    }
}



//             let val_inds' = map (adjustValInds n ns Ns val_inds) (iota Nmn)
//             let MO'' = scatter (replicate Nmn f32.nan) val_inds' MO'
//             in (MO'', MO', fst_break', mean)
void fstPComp(uint ns, uint Ns, int* val_inds, int* isBreak, int* fstBreak, int* adjBreak, int* fstBreakP) {
    // printf("\n\n\n---------------------- %d -----------------------\n\n\n", *isBreak);
    if (!isBreak){
        *fstBreakP = -1;
    } else {
        *adjBreak = adjustValInds(ns, Ns, val_inds, *fstBreak);
        *fstBreakP = ((*adjBreak-1) / 2) * 2 + 1;
    }

    if (ns <= 5 || Ns-ns <= 5) {
        *fstBreakP = -2;
    }
}


void valIndsPComp(uint Nmn, uint* ns, uint* Ns, int* fstBreak, int* val_inds, int* val_indsP) {
    for (int i = 0; i < Nmn; i++) {
        val_indsP[i] = adjustValInds(*ns, *Ns, val_inds, i);
    }
}


void MOppComp(uint Nmn, float* MOp, int* val_indsP, float* MOpp) {
     // printf("Nmn: %u\n", Nmn);
    for (int i = 0; i < Nmn; i++) {
        int currIdx = val_indsP[i];
        // if (currIdx != -1 || currIdx != -2) {
        // printf("1: currIdx: %u, MOpp: %f, MOp: %f\n",currIdx, MOpp[ currIdx ], MOp[i]);
        if (currIdx != -1 ) {
            MOpp[ currIdx ] = MOp[i];
            // printf("2: currIdx: %u, MOpp: %f, MOp: %f\n",currIdx, MOpp[ currIdx ], MOp[i]);
        }
    }
}


void ker10(float* bound, uint* Nss, uint* nss, float* sigmas, int* hs,
           float* MO_fsts, float* y_errors, int* val_indss, float* MOp,
           float* means, int* fstBreakP, float* MOpp){

    compBound(bound);
    uint Nmn = N-n;
    float* MO        = calloc(Nmn*m,sizeof(float));
    int* isBreak     = calloc(m,sizeof(int));
    int* fstBreak    = calloc(m,sizeof(int));
    int* adjBreak    = calloc(m,sizeof(int));
    int* val_indssP  = calloc(m*Nmn,sizeof(int));

    for (uint pix = 0; pix < m; pix++){
        // printf("\n---0----\n");
        MO_comp(Nmn, &hs[pix], &MO_fsts[pix], &y_errors[pix*N], &Nss[pix], &nss[pix], &MO[pix*Nmn]);
        // printf("\n---1----\n");
        MO_prime_comp(&MO[pix*Nmn], &nss[pix], sigmas[pix], &MOp[pix*Nmn]);

        breaks(&MOp[pix*Nmn], bound, Nss[pix], nss[pix], Nmn, &isBreak[pix], &fstBreak[pix]);

        meanComp(Nss[pix], nss[pix], Nmn, &MOp[pix*Nmn], &means[pix]);
        // printf("\n---4----\n");

        fstPComp(nss[pix], Nss[pix], &val_indss[pix*N], &isBreak[pix], &fstBreak[pix], &adjBreak[pix], &fstBreakP[pix]);
        // printf("\n---5----\n");

        valIndsPComp(Nmn, &nss[pix], &Nss[pix], &fstBreak[pix], &val_indss[pix*N], &val_indssP[pix*Nmn]);
        // printf("\n---6----\n");
        // printf("MOp: %f,  val_indssP: %u \n", MOp[pix*Nmn], val_indssP[pix*Nmn]);

        MOppComp(Nmn, &MOp[pix*Nmn], &val_indssP[pix*Nmn], &MOpp[pix*Nmn]);
        // printf("\n---7----\n");
    }

    free(MO);
    free(isBreak);
    free(fstBreak);
    free(adjBreak);
    free(val_indssP);
}


int main(int argc, char const *argv[]) {

	if (argc > 1) {
		printf("No arguments!\n");
      	return -1;
	}

	// printf("%d\n", trend);
	// printf("%d\n", k);
	// printf("%d\n", m);
	// printf("%d\n", n);
	// printf("%d\n", N);
	// printf("%f\n", freq);
	// printf("%f\n", hfrac);
	// printf("%f\n", lam);

    int K = 2*k +2;
    float* X  = calloc(K*N,sizeof(float));
    float* XT = calloc(K*N,sizeof(float));
    ker1(K,freq,X);
    transpose(K,X,XT);

    // printf("\n****** Printing X ******\n");
    printf("[");
    for (size_t i = 0; i < K; i++){
        printf("[");
        for (size_t j = 0; j < n; j++){
            uint index = i*N + j;
            printf(" %ff32", X[index]);
            if(i<=j-1){printf(",");}
        }
        printf("]");
        if(i<=K-1){printf(",");}
    }
    printf("]");


    // printf("\n****** Printing Y ******\n");
    // uint Ylen = sizeof(sample)/sizeof(sample[0]);
    // for (size_t i = 0; i < Ylen; i++){ // i < Ylen
    //     for (size_t j = 0; j < n; j++){
    //         printf(" %lf ", sample[i][j]);
    //     }
    //     printf("\n");
    // }

    // printf("\n****** Printing XT ******\n");
    // for (size_t i = 0; i < N; i++){
    //     for (size_t j = 0; j < K; j++){
    //         uint index = i*K + j;
    //         printf(" %f ", XT[index]);
    //     }
    //     printf("\n");
    // }

    // [n][m]
    float* Xsqr = calloc(K*K*m,sizeof(float));
    ker2(X, XT, Xsqr, K);

    // printf("\n****** Printing Xsqr ******\n");
    // for (int pix = 0; pix < m; pix++) {
    //     for (size_t i = 0; i < K; i++){
    //         for (size_t j = 0; j < K; j++){
    //             uint index = pix*K*K + i*K + j;
    //             printf("%f, ", Xsqr[index]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    float* XsqrInv = calloc(2*K*K*m,sizeof(float));
    ker3(Xsqr,XsqrInv,K);

    // printf("\n****** Printing XsqrInv ******\n");
    // for (int pix = 0; pix < m; pix++) {
    //     for (uint i = 0; i < K; i++){
    //         for (uint j = 0; j < 2*K; j++){
    //             uint index = pix*K*(2*K) + i*2*K + j;
    //             printf("%f, ", XsqrInv[index]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    float* XsqrInvLess = calloc(K*K*m,sizeof(float));
    doubleDown(XsqrInv, XsqrInvLess, K);

    // printf("\n****** Printing XsqrInvLess ******\n");
    // for (int pix = 0; pix < m; pix++) {
    //     for (size_t i = 0; i < K; i++){
    //         for (size_t j = 0; j < K; j++){
    //             uint index = pix*K*K + i*K + j;
    //             printf("%f, ", XsqrInvLess[index]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    float* B0 = calloc(K*m,sizeof(float));
    ker4(X, K, B0);

    // printf("\n****** Printing B0 ******\n");
    // for (uint i = 0; i < m; i++){
    //     for (int j = 0; j < K; j++) {
    //         uint index = i*K + j;
    //         printf("%f, ", B0[index]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    float* B = calloc(K*m,sizeof(float));
    ker5(XsqrInvLess, K, B0, B);

    // printf("\n****** Printing B ******\n");
    // for (uint i = 0; i < m; i++){
    //     for (int j = 0; j < K; j++) {
    //         uint index = i*K + j;
    //         printf("%f, ", B[index]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    float* yhat = calloc(m*N,sizeof(float));
    ker6(XT, B, K, yhat);

    // printf("\n****** Printing yhat ******\n");
    // for (uint i = 0; i < m; i++){
    //     for (int j = 0; j < N; j++) {
    //         uint index = i*N + j;
    //         printf("%f, ", yhat[index]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    uint* Nss           = calloc(m,sizeof(uint));
    float* y_errors_all = calloc(m*N,sizeof(float));
    int* val_indss      = calloc(m*N,sizeof(uint));
    float* y_errors     = calloc(m*N,sizeof(float));

    for (int i = 0; i < m*N; i++) { y_errors[i] = F32_MIN; }

    ker7(yhat, y_errors_all, Nss, y_errors, val_indss);

    // printf("\n****** Printing Nss ******\n");
    // for (uint i = 0; i < m; i++){
    //     printf("%d, ", Nss[i]);
    // }
    // printf("\n");

    // printf("\n****** Printing y_errors ******\n");
    // for (uint i = 0; i < m; i++){
    //     for (int j = 0; j < N; j++) {
    //         uint index = i*N + j;
    //         printf("%f, ", y_errors[index]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    // printf("\n****** Printing val_indss ******\n");
    // for (uint i = 0; i < m; i++){
    //     for (int j = 0; j < N; j++) {
    //         uint index = i*N + j;
    //         printf("%d, ", val_indss[index]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    int* hs       = calloc(m,sizeof(int));
    uint* nss     = calloc(m,sizeof(uint));
    float* sigmas = calloc(m,sizeof(float));
    ker8(y_errors, K, hs, nss, sigmas);

    // printf("\n****** Printing hs ******\n");
    // for (uint i = 0; i < m; i++){
    //     printf("%d, ", hs[i]);
    // }
    // printf("\n");

    // printf("\n****** Printing nss ******\n");
    // for (uint i = 0; i < m; i++){
    //     printf("%u, ", nss[i]);
    // }
    // printf("\n");

    // printf("\n****** Printing sigmas ******\n");
    // for (uint i = 0; i < m; i++){
    //     printf("%f, ", sigmas[i]);
    // }
    // printf("\n");

    float* MO_fsts = calloc(m,sizeof(float));
    ker9(hs, y_errors, nss, MO_fsts);

    // printf("\n****** Printing MO_fsts ******\n");
    // for (uint i = 0; i < m; i++){
    //     printf("%f, ", MO_fsts[i]);
    // }
    // printf("\n");

    float* bound   = calloc(N-n,sizeof(float));
    float* MOp     = calloc(m*(N-n),sizeof(float));
    float* means   = calloc(m,sizeof(float));
    int* fstBreakP = calloc(m,sizeof(int));
    float* MOpp    = calloc(m*(N-n),sizeof(float));

    for (int i = 0; i < m*(N-n); i++) { MOpp[i] = F32_MIN; }

    ker10(bound, Nss, nss, sigmas, hs, MO_fsts, y_errors, val_indss, MOp, means, fstBreakP, MOpp);

    // printf("\n****** Printing MOpp ******\n");
    // for (uint i = 0; i < m; i++){
    //     for (int j = 0; j < N-n; j++) {
    //         uint index = i*(N-n) + j;
    //         printf("%f, ", MOpp[index]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    // printf("\n****** Printing MOp ******\n");
    // for (uint i = 0; i < m; i++){
    //     for (int j = 0; j < N-n; j++) {
    //         uint index = i*(N-n) + j;
    //         printf("%f, ", MOp[index]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    // printf("\n****** Printing fstBreakP ******\n");
    // for (uint i = 0; i < m; i++){
    //     printf("%d, ", fstBreakP[i]);
    // }
    // printf("\n");

    // printf("\n****** Printing means ******\n");
    // for (uint i = 0; i < m; i++){
    //     printf("%f, ", means[i]);
    // }
    // printf("\n");


    free(X);
    free(XT);
    free(Xsqr);
    free(XsqrInv);
    free(B0);
    free(B);
    free(yhat);
    free(y_errors_all);
    free(Nss);
    free(y_errors);
    free(val_indss);
    free(hs);
    free(nss);
    free(sigmas);
    free(MO_fsts);
    free(bound);
    free(MOp);
    free(means);
    free(fstBreakP);

	return 0;
}



