all: poisson

# -g outputs debugging information
# -Wall enables all warnings
# -pthread configures threading
CC = g++
CFLAGS = -g -Wall -pthread -std=c++11

poisson: poisson.cpp poisson_solver.cpp
	$(CC) -o $@ $^ $(CFLAGS)

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
