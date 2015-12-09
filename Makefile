# Unix makefile

CC=clang
CFLAGS=-Wall -pedantic
LDFLAGS=-Bstatic -lstdc++ -lOpenCL

.PHONY: dagSimCL

dagSimCL: clean release

release: dagSimCL.cpp
	$(CC) -O2 $^ $(LDFLAGS) -o dagSimCL-linux64
	strip dagSimCL-linux64

debug: dagSimCL.cpp
	$(CC) $(CFLAGS) -g -O0 $^ $(LDFLAGS) -o dagSimCL-linux64-debug

clean:
	rm -rf *.o
	rm -rf *.out
	rm -rf dagSimCL-linux64
	rm -rf dagSimCL-linux64-debug
