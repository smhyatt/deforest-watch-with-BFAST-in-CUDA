CXX = nvcc

SRC = main-seq.cu
HELPERS    = helper.cu.h kernels-naive.cu.h sequential.cu.h
EXECUTABLE = bfast-project

default: compile run

.cu.o: $(SRC) $(HELPERS)
	$(CXX) -c $@ $<

bfast: $(EXECUTABLE)
		./$(EXECUTABLE)

compile: $(EXECUTABLE)

$(EXECUTABLE): $(SRC) $(HELPERS)
	$(CXX) -o $(EXECUTABLE) $(SRC)

run: $(EXECUTABLE)
	./$(EXECUTABLE) sahara

seqValidering:
	futhark c ../bfast-irreg.fut
	./bfast-irreg < ../data/testset_sahara_2pix.in > ../data/val.data
	$(CC) $(CFLAGS) seq.c && ./a.out >> ../data/val.data
	futhark c ../validation-benchmark.fut
	./validation-benchmark < ../data/val.data
	rm validation-benchmark
	rm validation-benchmark.c
	rm bfast-irreg
	rm a.out

clean:
	rm -f $(EXECUTABLE)

