
# Notes and commands

```
gzip -d < sahara.in.gz > sahara.in
```

```
gzip -d < peru.in.gz > peru.in
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
