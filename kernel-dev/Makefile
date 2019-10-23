CXX = nvcc

SRC1 = main-seq.cu
HELPERS    = helper.cu.h kernels-naive.cu.h sequential.cu.h
EXECUTABLE = bfast-project

default: compile run

.cu.o: $(SRC1) $(HELPERS)
	$(CXX) -c $@ $<

bfast: $(EXECUTABLE)
		./$(EXECUTABLE)

compile: $(EXECUTABLE)

$(EXECUTABLE): $(SRC1) $(HELPERS)
	$(CXX) -o $(EXECUTABLE) $(SRC1)

run: $(EXECUTABLE)
	./$(EXECUTABLE) sahara

seqValidering:
	futhark c ../bfast-irreg.fut
	./bfast-irreg < ../data/testset_sahara_2pix.in > ../data/val.data
	run
	futhark c ../validation-benchmark.fut
	./validation-benchmark < ../data/val.data
	rm validation-benchmark
	rm validation-benchmark.c
	rm bfast-irreg


clean:
	rm -f $(EXECUTABLE)

