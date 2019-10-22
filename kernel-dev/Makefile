CXX = nvcc

SRC = main.cu
HELPERS    = helper.cu.h kernels.cu.h sequential.cu.h
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
	./$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE) 

