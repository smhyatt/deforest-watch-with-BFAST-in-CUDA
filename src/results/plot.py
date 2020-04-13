import numpy as np
import matplotlib.pyplot as plt

# plt.plot([1,2,3,4])
# plt.ylabel('some numbers')
# plt.show()


# # evenly sampled time at 200ms intervals
# t = np.arange(0., 5., 0.2)

# # red dashes, blue squares and green triangles
# plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
# plt.show()


# PRIMES 

# Checking performance of primes-naive.fut with C and 10000000 numbers.
# futhark c primes-naive.fut
# echo "10000000" | ./primes-naive -t /dev/stderr -r 10 > /dev/null
naivc = [625269, 624408, 625119, 623531, 624216, 624373, 624946, 623653, 626097, 623879]

# Checking performance of primes-flat.fut with C and 10000000 numbers.
# futhark c primes-flat.fut
# echo "10000000" | ./primes-flat -t /dev/stderr -r 10 > /dev/null
flatc = [356735, 354949, 355290, 356287, 357054, 355508, 355659, 355381, 355542, 356251]

# Checking performance of primes-seq.fut with C and 10000000 numbers.
# futhark c primes-seq.fut
# echo "10000000" | ./primes-seq -t /dev/stderr -r 10 > /dev/null
seqc = [312557, 312997, 311324, 311968, 311628, 312992, 316711, 314645, 311046, 311160]

# Checking performance of primes-naive.fut with OpenCL and 10000000 numbers.
# futhark opencl primes-naive.fut
# echo "10000000" | ./primes-naive -t /dev/stderr -r 10 > /dev/null
naivcl = [32625, 32007, 33381, 31230, 29731, 29848, 29912, 31121, 31931, 32142]

# Checking performance of primes-flat.fut with OpenCL and 10000000 numbers.
# futhark opencl primes-flat.fut
# echo "10000000" | ./primes-flat -t /dev/stderr -r 10 > /dev/null
flatcl = [20095, 19752, 19264, 20747, 19913, 19448, 20358, 20055, 19737, 19370]


X2 = np.arange(0, 10, 1)

plt.title('Time differences in running Prime-Numbers Computation sequentially \n and parallel with both a flat and not-flat implementation.') 
# plt.xscale('log')
# plt.yscale('log')
plt.xlabel('Ten test runs.')
plt.ylabel('The runtime.') 

# plt.plot(X2, naivc, linestyle='-', label = "primes-naive w/ C")
plt.plot(X2, naivcl, linestyle='-.', label = "primes-naive w/ OpenCL")

# plt.plot(X2, flatc, linestyle='-', label = "primes-flat w/ C")
plt.plot(X2, flatcl, linestyle='-.', label = "primes-flat w/ OpenCL")

# plt.plot(X2, seqc, linestyle='-', label = "primes-seq w/ C")

plt.legend()
plt.show()











# LSSP

seqsorted = [19955, 19940, 20028, 19902, 19933, 19900, 19902, 19978, 19929, 19906]
parsorted = [2784, 2838, 2785, 2780, 2824, 2789, 2777, 2779, 2796, 2792]

seqzeros = [23673, 23707, 23704, 23700, 23737, 23678, 23663, 23706, 23730, 23718]
parzeros = [2774, 2768, 2816, 2771, 2770, 2768, 2775, 2769, 2778, 2778]

seqsame = [19944, 19957, 20154, 19901, 19932, 19898, 19900, 19898, 19960, 19902]
parsame = [2788, 2788, 2787, 2784, 2787, 2781, 2775, 2792, 2786, 2776]

X1 = np.arange(0, 10000, 1000)

plt.title('Time differences in running Longest \n Satisfying Segment sequentially and parallel.') 
# plt.xscale('log')
# plt.yscale('log')
plt.xlabel('Ten test runs.')
plt.ylabel('The runtime.') 

plt.plot(X1, seqsorted, linestyle='-', label = "lssp-sorted - CPU")
# plt.plot(X1, parsorted, linestyle='-.', label = "lssp-sorted - GPU")
plt.plot(X1, seqzeros, linestyle='-', label = "lssp-zeros - CPU")
# plt.plot(X1, parzeros, linestyle='-.', label = "lssp-zeros - GPU")
plt.plot(X1, seqsame, linestyle='-', label = "lssp-same - CPU")
# plt.plot(X1, parsame, linestyle='-.', label = "lssp-same - GPU")
plt.legend()
plt.show()




# CUDA 

arrsze = [100, 300, 500, 1000, 50000, 100000, 500000, 753411, 5000000, 10000001, 50000000, 100000001, 1000000001] 
# arrsze = [500, 753411, 10000001, 1000000001] 
# arrsze = np.arange(0, 1000000000, 250000000)
# cputime = [6, 10349, 139112, 13545874]
# gputime = [6, 101, 1159, 68]
cputime = [1, 4, 6, 13, 694, 1356, 6779, 10349, 67729, 139112, 676481, 1361891, 13545874]
gputime = [6, 6, 6, 7, 11, 18, 71, 101, 570, 1159, 5110, 9997, 68]

plt.title('Time differences in running CUDA program.') 
plt.xscale('log')
plt.yscale('log')
plt.xlabel('The sizes of the arrays in log.')
plt.ylabel('The runtime in log.') 

plt.plot(arrsze, cputime, linestyle='-', label = "CPU")
plt.plot(arrsze, gputime, linestyle='-.', label = "GPU")
plt.legend()
plt.show()


def Average(lst): 
    return sum(lst) / len(lst)


# SPARSE VECTOR
# X = np.arange(0, 1100, 110)
X = np.arange(0, 10, 1)
seq = [1066, 1002, 1000, 990, 1033, 994, 989, 988, 1030, 989]
flat = [600, 608, 580, 601, 488, 576, 597, 596, 597, 593]

plt.title('Time differences in running Sparse-Matrix \n Vector Multiplication sequential and flat.') 
# plt.xscale('log')
# plt.yscale('log')
plt.xlabel('The ten dataset.')
plt.ylabel('The runtime.') 

plt.plot(X, seq, linestyle='-', label = "CPU")
plt.plot(X, flat, linestyle='-.', label = "GPU")
plt.legend()
plt.show()


averageseq = Average(seq)
averageflat = Average(flat)

print("Average of the list =", round(averageseq, 2), round(averageflat, 2)) 



# ./a.out 1000

# CPU took 13 microseconds (0.01ms)

# GPU took 7 microseconds (0.01ms)



# ./a.out 50000

# CPU took 694 microseconds (0.69ms)

# GPU took 11 microseconds (0.01ms)




# ./a.out 100000

# CPU took 1356 microseconds (1.36ms)

# GPU took 18 microseconds (0.02ms)


# ./a.out 500000

# CPU took 6779 microseconds (6.78ms)

# GPU took 71 microseconds (0.07ms)



# ./a.out 5000000

# CPU took 67729 microseconds (67.73ms)

# GPU took 570 microseconds (0.57ms)


# [500, 1000, 50000, 100000, 500000, 753411, 5000000, 10000001, 50000000, 100000001, 1000000001] 

# cputime = [6, 13, 694, 1356, 6779, 10349, 67729, 139112, 676481, 1361891, 13545874]
# gputime = [6, 7, 11, 18, 71, 101, 570, 1159, 5110, 9997, 68]


# ./a.out 50000000

# CPU took 676481 microseconds (676.48ms)

# GPU took 5110 microseconds (5.11ms)



# ./a.out 100000001

# CPU took 1361891 microseconds (1361.89ms)

# GPU took 9997 microseconds (10.00ms)




# 753411
# 10349
# 101

# 500
# 6
# 6

# 10000001
# 139112
# 1159

# 1000000001
# 13545874
# 68


