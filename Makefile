# Unix makefile

CC=g++
CFLAGS=-Wall -pedantic -O2 ###-g -DDEBUG -O0
LDFLAGS=-lstdc++ -lOpenCL

.PHONY: clean

dagSimCL: dagSimCL.cpp
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@
	strip $@

clean:
	rm -rf *.o
	rm -rf dagSimCL
