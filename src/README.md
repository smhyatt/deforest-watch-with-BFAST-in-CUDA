

# This is a Guide to Run and Validate our Implementation of BFAST

## Initial Set-Up Step

1. If you haven't run anything before, start by decompressing and creating the datasets ready for parsing (THIS STEP SHOULD ONLY BE APPLIED ONCE). 
    1. `cd parser && make decompress && make parse && cd ..`

2. Remember to modify the bfast-irreg.fut and validation.fut to get the output you want to test; both are in the validation folder. 

3. To run and validate you have to work from the kernels folder
    1. `cd kernels`

## Compiling and Running without Validation 

If you only want to compile to get the performance without validation, you can run the following.

### Peru Dataset with the Sequential Implementation:

Running the sequential implementation on the full dataset 
```make PseqFullRun```


Running the sequential implementation on the test-set 
```make P2seqFullRun```


### Peru Dataset with the Optimised Implementation:

Running the optimised implementation on the full dataset 
```make PopFullRun```


Running the optimised implementation on the test-set 
```make P2opFullRun```



### Sahara Dataset with the Sequential Implementation:

Running the sequential implementation on the full dataset 
```make SseqFullRun```


Running the sequential implementation on the test-set 
```make S2seqFullRun```


### Sahara Dataset with the Optimised Implementation:

Running the optimised implementation on the full dataset 
```make SopFullRun```


Running the optimised implementation on the test-set 
```make S2opFullRun```


## Compiling and Running with Validation

### Peru Optimized on the full set:

```make peru```


### Peru Optimized on the test-set:
```make perutest```


### Peru Sequential on the full set:
```make peruseq```


### Peru Sequential on the test-set:
```make perutestseq```


### Sahara Optimized on the full set:
```make sahara```


### Sahara Optimized on the test-set:
```make saharatest```


### Sahara Sequential on the full set:
```make saharaseq```


### Sahara Sequential on the test-set:
```make saharatestseq```



## Compiling and Running the Sequential Version Against the Parallel one, with validation

### Sahara:
```make sval```


### Peru: 
```make pval```




