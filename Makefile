CC = gcc
CFLAGS = -O3 -fomit-frame-pointer -std=c11 -Wall -Wextra -pedantic -g
INC = -I src
LIB = -L lib

CUDA_LIB  ?= -L /opt/cuda/lib64

CC_LINK = -lm -lcuda -lcudart -lstdc++

CSVCONVERT = build/csvconvert.o

KMEANS  = build/kmeans.o build/datafile.o build/kmeans_cpu_impl.o
KMEANS += build/kmeans_gpu.o

all: kmeans csvconvert

kmeans: $(KMEANS)
	$(CC) $(CFLAGS) $(LIB) -o $@ $^ $(CUDA_LIB) $(CC_LINK)

csvconvert: $(CSVCONVERT)
	$(CC) $(CFLAGS) $(LIB) -o $@ $^

build/%.o: src/%.c
	mkdir -p build
	$(CC) -c $(CFLAGS) $(INC) $< -o $@

build/%.o: src/%.cu
	mkdir -p build
	nvcc $< -o $@ -lcuda -c $(INC)

clean:
	rm -rf build
	rm -f csvconvert
