

# This is a Guide to Run and Validate BFAST

# 1. If you haven't run anything before, start by decompressing and creating the datasets ready for parsing (THIS STEP SHOULD ONLY BE APPLIED ONCE). 

```
cd parser && make decompress && make parse && cd ..
```

# 2. Remember to modify the bfast-irreg.fut and validation.fut to get the output you want to test, both are in the validation folder. 


# 3. To run and validate you have to work from the kernels folder
```
cd kernels
```

## 3.1 Compile and run without validation 

If you only want to compile to get the performance without validation you can run the following.

### Peru Sequentail: 

Running the sequential implementation on the full dataset 
```make PseqFullRun```


Running the sequential implementation on the testset 
```make P2seqFullRun```


### Peru Optimized: 

Running the optimized implementation on the full dataset 
```make PopFullRun```


Running the optimized implementation on the testset 
```make P2opFullRun```



### Sahara Sequentail

Running the sequential implementation on the full dataset 
```make SseqFullRun```


Running the sequential implementation on the testset 
```make S2seqFullRun```


### Sahara Optimized

Running the optimized implementation on the full dataset 
```make SopFullRun```


Running the optimized implementation on the testset 
```make S2opFullRun```


## 3.2 Compile and run with validation 

### Peru Optimized on the full set:

```make peru```


### Peru Optimized on the testset:
```make perutest```


### Peru Sequentail on the full set:
```make peruseq```


### Peru Sequentail on the testset:
```make perutestseq```


### Sahara Optimized on the full set:
```make sahara```


### Sahara Optimized on the testset:
```make saharatest```


### Sahara Sequentail on the full set:
```make saharaseq```


### Sahara Sequentail on the testset:
```make saharatestseq```



## 3.1 Compile and run the sequential version against the parallel one, with validation 

### Sahara:
```make sval```


### Peru: 
```make pval```















