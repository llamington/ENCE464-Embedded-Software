all: poisson

# -g outputs debugging information
# -Wall enables all warnings
# -pthread configures threading
CXX = g++
CXXFLAGS = -g -Wall -pthread -std=c++11 -lstdc++

poisson: poisson.cpp poisson_solver.cpp
	$(CXX) -o $@ $^ $(CXXFLAGS)

.PHONY: disassembly
disassembly: poisson.s

poisson.s: poisson
	objdump -S --disassemble $< > $@

.PHONY: test
test: poisson
	./test.sh

.PHONY: clean
clean:
	rm -f poisson *.o *.s
