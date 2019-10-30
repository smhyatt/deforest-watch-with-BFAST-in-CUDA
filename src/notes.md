
# Notes and commands

```
gzip -d < sahara.in.gz > sahara.in
```

```
futhark bench --backend=opencl bfast-irreg.fut
Compiling bfast-irreg.fut...
Results for bfast-irreg.fut:
dataset data/sahara.in: bfast-irreg.actual and bfast-irreg.expected do not match:
Value (0,3) expected 97i32, got 3i32
➜  pmph/src/pimph git:(master) ✗ futhark bench --backend=c bfast-irreg.fut
Compiling bfast-irreg.fut...
Results for bfast-irreg.fut:
dataset data/sahara.in: 12331527.80μs (avg. of 10 runs; RSD: 0.01)
```

### sequential 
For Xsqr da epsilon var 0.01 (nu ændret til 0.1)
false
4471 i32
0.000131f32
-0.000015f32


For Xsqr da cut off (nr. 2 epsilon) var 0.01 (nu ændret til 0.1)
false
11086 i32
-0.006286f32
-0.001464f32


false
109133 i32
-0.012177f32
-0.001053f32


### optimized 

For Xsqr da epsilon og cutoff er 0.1
false
109133 i32
-0.014771f32
-0.001030f32

sahara fuld sæt m = 67968, N = 414

67968 \* 414 = 28.138.752

109133/414 = 263

<!-- 28138752 / 109133 = 257,8390770894
257,8390770894 / 414 = 0,62 -->

263/30 = 8,7666

30 pixler pr. block

peru fuld sæt m = 111556, N = 235


X for peru, eps1 = 0,01 eps2 = 0,001 
false
1522i32
0.000016f32
-0.000000f32


false
29312 i32
1846861 i32
0.403137 f32
0.496277 f32


false
29276 i32
3169677 i32
-0.815186 f32
-0.686035 f32





