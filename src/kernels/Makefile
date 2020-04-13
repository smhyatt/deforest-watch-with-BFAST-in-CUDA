# KERNELS FOLDER
# MAKEFILE FOR RUNNING THE KERNELS AND VALIDATING THE OUTPUT

CXX = nvcc -ftz=false -prec-div=true -prec-sqrt=true

SRC1 = main-seq.cu
SRC2 = main-naive.cu
SRC3 = main-optim.cu
SOURCES_CPP = main-seq.cu main-naive.cu main-optim.cu
HELPERS     = helper.cu.h kernels-naive.cu.h kernels-optim.cu.h sequential.cu.h
EXECUTABLE1 = seq-bfast-project
EXECUTABLE2 = naive-bfast-project
EXECUTABLE3 = optim-bfast-project



#######################################################################
############################## COMPILING ##############################
#######################################################################

# Compiling the sequential implementation
compileSeq: $(EXECUTABLE1)

$(EXECUTABLE1): $(SRC1) $(HELPERS)
	$(CXX) -o $(EXECUTABLE1) $(SRC1)

# Compiling the naive implementation
compileNaive: $(EXECUTABLE3)

$(EXECUTABLE2): $(SRC2) $(HELPERS)
	$(CXX) -o $(EXECUTABLE2) $(SRC2)


# Compiling the optimized implementation
compileOP: $(EXECUTABLE3)

$(EXECUTABLE3): $(SRC3) $(HELPERS)
	$(CXX) -o $(EXECUTABLE3) $(SRC3)


# Compile sequential, naive and optimized implementations
compile: $(EXECUTABLE1) $(EXECUTABLE2) $(EXECUTABLE3)

$(EXECUTABLE1): $(SRC1) $(HELPERS)
	$(CXX) -o $(EXECUTABLE1) $(SRC1)

$(EXECUTABLE2): $(SRC2) $(HELPERS)
	$(CXX) -o $(EXECUTABLE2) $(SRC2)

$(EXECUTABLE3): $(SRC3) $(HELPERS)
	$(CXX) -o $(EXECUTABLE3) $(SRC3)



####################################################################
############################## SAHARA ##############################
####################################################################

############### RUN AND NO VALIDATION ###############

# FULL SET
# Running the sequential implementation on the full dataset 
SseqFullRun: $(EXECUTABLE1)
			./$(EXECUTABLE1) sahara

# TESTSET
# Running the sequential implementation on the testset 
S2seqFullRun: $(EXECUTABLE1)
			./$(EXECUTABLE1) 2sahara


# FULL SET
# Running the optimized implementation on the full dataset 
SopFullRun: $(EXECUTABLE3)
			./$(EXECUTABLE3) sahara

# TESTSET
# Running the optimized implementation on the testset 
S2opFullRun: $(EXECUTABLE3)
			./$(EXECUTABLE3) 2sahara





############### RUN AND VALIDATION ###############

# FULL SET
# Runs and validates the sequential implementation of the full Sahara dataset  

seqVsahara: 
	$(info )
	$(info The validation for the sequential implementation with the full Sahara dataset.)
	# creates the validation results from bfast-irreg 
	cd ../validation && make fullSeqSahara && cd ../kernels
	# compile and run sequential implementation with full dataset
	make compileSeq
	# the run below inserts the data in the correct validation file
	./$(EXECUTABLE1) sahara
	# compiling validation file
	cd ../validation && futhark c validation.fut
	# validating the bfast result with our sequential implementation
	cd ../validation && ./validation < ../data/saharaval.data
	# clean-up 
	cd ../validation && rm validation validation.c || true


# TESTSET
# Runs and validates the sequential implementation of two pixels from the Sahara dataset  

seq2Vsahara: 
	$(info )
	$(info The validation for the sequential implementation with two pixels of the Sahara dataset.)
	cd ../validation && make 2pixelsSeq && cd ../kernels
	# compile and run sequential implementation with test dataset
	make compileSeq
	# the run below inserts the data in the correct validation file
	./$(EXECUTABLE1) 2sahara
	# compiling validation file
	cd ../validation && futhark c validation.fut
	# validating the bfast result with our sequential implementation
	cd ../validation && ./validation < ../data/sahara2val.data
	# clean-up 
	cd ../validation && rm validation validation.c || true


# FULL SET
# Runs and validates the optimized implementation of the full Sahara dataset  

opVsahara: 
	$(info )
	$(info The validation for the optimized implementation with the full Sahara dataset.)
	# creates the validation results from bfast-irreg 
	cd ../validation && make fullParSahara && cd ../kernels
	# compile and run optimized implementation with full dataset
	make compileOP
	# the run below inserts the data in the correct validation file
	./$(EXECUTABLE3) sahara
	# compiling validation file
	cd ../validation && futhark opencl validation.fut
	# validating the bfast result with our sequential implementation
	cd ../validation && ./validation < ../data/saharaval.data
	# clean-up 
	cd ../validation && rm validation validation.cl || true 


# TESTSET
# Runs and validates the optimized implementation of two pixels from the Sahara dataset  

op2Vsahara: 
	$(info )
	$(info The validation for the optimized implementation with two pixels of the Sahara dataset.)
	cd ../validation && make 2pixelsPar && cd ../kernels
	# compile and run optimized implementation with test dataset
	make compileOP
	# the run below inserts the data in the correct validation file
	./$(EXECUTABLE3) 2sahara
	# compiling validation file
	cd ../validation && futhark opencl validation.fut
	# validating the bfast result with our sequential implementation
	cd ../validation && ./validation < ../data/sahara2val.data
	# clean-up 
	cd ../validation && rm validation validation.cl || true




####################################################################
##############################  PERU  ##############################
####################################################################


############### RUN AND NO VALIDATION ###############

# FULL SET
# Running the sequential implementation on the full dataset 
PseqFullRun: $(EXECUTABLE1)
			./$(EXECUTABLE1) peru

# TESTSET
# Running the sequential implementation on the testset 
P2seqFullRun: $(EXECUTABLE1)
			./$(EXECUTABLE1) 2peru


# FULL SET
# Running the optimized implementation on the full dataset 
PopFullRun: $(EXECUTABLE3)
			./$(EXECUTABLE3) peru

# TESTSET
# Running the optimized implementation on the testset 
P2opFullRun: $(EXECUTABLE3)
			./$(EXECUTABLE3) 2peru



############### RUN AND VALIDATION ###############

# FULL SET
# Runs and validates the sequential implementation of the full Peru dataset  

seqVperu: 
	$(info )
	$(info The validation for the sequential implementation with the full Peru dataset.)
	# creates the validation results from bfast-irreg 
	cd ../validation && make fullSeqPeru && cd ../kernels
	# compile and run sequential implementation with full dataset
	make compileSeq
	# the run below inserts the data in the correct validation file
	./$(EXECUTABLE1) peru
	# compiling validation file
	cd ../validation && futhark c validation.fut
	# validating the bfast result with our sequential implementation
	cd ../validation && ./validation < ../data/peruval.data
	# clean-up 
	cd ../validation && rm validation validation.c || true


# TESTSET
# Runs and validates the sequential implementation of two pixels from the Peru dataset  

seq2Vperu: 
	$(info )
	$(info The validation for the sequential implementation with two pixels of the Peru dataset.)
	cd ../validation && make 2pixelsSeq && cd ../kernels
	# compile and run sequential implementation with test dataset
	make compileSeq
	# the run below inserts the data in the correct validation file
	./$(EXECUTABLE1) 2peru
	# compiling validation file
	cd ../validation && futhark c validation.fut
	# validating the bfast result with our sequential implementation
	cd ../validation && ./validation < ../data/peru2val.data
	# clean-up 
	cd ../validation && rm validation validation.c || true


# FULL SET
# Runs and validates the optimized implementation of the full Peru dataset  

opVperu: 
	$(info )
	$(info The validation for the optimized implementation with the full Peru dataset.)
	# creates the validation results from bfast-irreg 
	cd ../validation && make fullParPeru && cd ../kernels
	# compile and run optimized implementation with full dataset
	make compileOP
	# the run below inserts the data in the correct validation file
	./$(EXECUTABLE3) peru
	# compiling validation file
	cd ../validation && futhark opencl validation.fut
	# validating the bfast result with our sequential implementation
	cd ../validation && ./validation < ../data/peruval.data
	# clean-up 
	cd ../validation && rm validation validation.cl || true 


# TESTSET
# Runs and validates the optimized implementation of two pixels from the Peru dataset  

op2Vperu: 
	$(info )
	$(info The validation for the optimized implementation with two pixels of the Peru dataset.)
	cd ../validation && make 2pixelsPar && cd ../kernels
	# compile and run optimized implementation with test dataset
	make compileOP
	# the run below inserts the data in the correct validation file
	./$(EXECUTABLE3) 2peru
	# compiling validation file
	cd ../validation && futhark opencl validation.fut
	# validating the bfast result with our sequential implementation
	cd ../validation && ./validation < ../data/peru2val.data
	# clean-up 
	cd ../validation && rm validation validation.cl || true




#############################################################################
##############################  FAST COMMANDS  ##############################
#############################################################################

# PERU OPTIMIZED FULL SET
peru: opVperu

# PERU OPTIMIZED TESTSET
perutest: op2Vperu

# PERU SEQUENTIAL FULL SET
peruseq: seqVperu

# PERU SEQUENTIAL TESTSET
perutestseq: seq2Vperu


# SAHARA OPTIMIZED FULL SET 
sahara:	opVsahara

# SAHARA OPTIMIZED TESTSET
saharatest: op2Vsahara

# SAHARA SEQUENTIAL FULL SET
saharaseq: seqVsahara

# SAHARA SEQUENTIAL TESTSET
saharatestseq: seq2Vsahara



########################################################################################
##############################  VALIDATING SEQ. W/ PAR.   ##############################
########################################################################################



# Validation of the sequential implementation against the parallel one with Sahara

sval:
	# empties the data file
	rm ../data/saharaval.data || true
	# compiles the sequential 
	make compileSeq
	# compiles the optimized
	make compileOP
	# runs the sequential and adds to ../data/saharaval.data
	make SseqFullRun
	# runs the optimized and adds to ../data/saharaval.data
	make SopFullRun
	# compiling validation file
	cd ../validation && futhark opencl validation.fut
	# runs a validation against sequential and optimized
	cd ../validation && ./validation < ../data/saharaval.data
	# clean-up 
	cd ../validation && rm validation || true



# Validation of the sequential implementation against the parallel one with Peru

pval:
	# empties the data file
	rm ../data/peruval.data || true
	# compiles the sequential 
	make compileSeq
	# compiles the optimized
	make compileOP
	# runs the sequential and adds to ../data/saharaval.data
	make PseqFullRun
	# runs the optimized and adds to ../data/saharaval.data
	make PopFullRun
	# compiling validation file
	cd ../validation && futhark opencl validation.fut
	# runs a validation against sequential and optimized
	cd ../validation && ./validation < ../data/peruval.data
	# clean-up 
	cd ../validation && rm validation || true
















