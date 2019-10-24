CXX = nvcc

SRC1 = main-seq.cu
SRC2 = main-naive.cu
SRC3 = main-optim.cu
SOURCES_CPP = main-seq.cu main-naive.cu main-optim.cu
HELPERS     = helper.cu.h kernels-naive.cu.h kernels-optim.cu.h sequential.cu.h
EXECUTABLE1 = seq-bfast-project
EXECUTABLE2 = naive-bfast-project
EXECUTABLE3 = optim-bfast-project

default: compile run

.cu.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) -c $@ $<


seqbfast: $(EXECUTABLE1)
			./$(EXECUTABLE1) sahara

naivebfast: $(EXECUTABLE2)
			./$(EXECUTABLE2) sahara

optimbfast: $(EXECUTABLE3)
			./$(EXECUTABLE3) sahara


compile: $(EXECUTABLE1) $(EXECUTABLE2) $(EXECUTABLE3)

$(EXECUTABLE1): $(SRC1) $(HELPERS)
	$(CXX) -o $(EXECUTABLE1) $(SRC1)

$(EXECUTABLE2): $(SRC2) $(HELPERS)
	$(CXX) -o $(EXECUTABLE2) $(SRC2)

$(EXECUTABLE3): $(SRC3) $(HELPERS)
	$(CXX) -o $(EXECUTABLE3) $(SRC3)



run: $(EXECUTABLE1) $(EXECUTABLE2) $(EXECUTABLE3)
	./$(EXECUTABLE1) sahara
	./$(EXECUTABLE2) sahara
	./$(EXECUTABLE3) sahara


futV:
	futhark c ../bfast-irreg.fut
	futhark c ../validation-benchmark.fut

clear:
	.././bfast-irreg < ../data/testset_sahara_2pix.in > ../data/val.data

seqV: clear seqbfast
	$(info )
	$(info The validation for the sequential implementation.)
# 	cat ../data/val.data
	.././validation-benchmark < ../data/val.data
	rm ../data/val.data

naiveV: clear naivebfast
	$(info )
	$(info The validation for the naive implementation.)
	.././validation-benchmark < ../data/val.data
	rm ../data/val.data

optimV: clear optimbfast
	$(info )
	$(info The validation for the optimized implementation.)
	.././validation-benchmark < ../data/val.data
# 	cat ../data/val.data
	rm ../data/val.data

fullV: compile futV optimV seqV naiveV 
	rm ../validation-benchmark
	rm ../validation-benchmark.c
	rm ../bfast-irreg


clean:
	rm -f $(EXECUTABLE1) $(EXECUTABLE2) $(EXECUTABLE3)






