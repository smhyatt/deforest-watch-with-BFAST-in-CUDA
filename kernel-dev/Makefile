CXX = nvcc

SRC = main.cu
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

clean:
	rm -f $(EXECUTABLE)

