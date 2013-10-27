CC = gcc
CFLAGS = -O3 -std=c99 -Wall -Wextra -pedantic -fomit-frame-pointer
INC = -I src
LIB = -L lib

CUDA_LIB  ?= /opt/cuda/lib64
CUDA_INC  ?= /opt/cuda/include
CUDA_FLAGS = 

NVCC ?= nvcc

CC_LINK = -lm -lcuda -lcudart -lstdc++

CSVCONVERT = build/csvconvert.o

KMEANS  = build/kmeans.o build/datafile.o build/kmeans_cpu_impl.o
KMEANS += build/kmeans_gpu.o

all: kmeans csvconvert

debug: CUDA_FLAGS += --ptxas-options=-v -g -G
debug: CFLAGS     += -g
debug: kmeans csvconvert

kmeans: $(KMEANS)
	$(CC) $(CFLAGS) $(LIB) -o $@ $^ -L $(CUDA_LIB) $(CC_LINK)

csvconvert: $(CSVCONVERT)
	$(CC) $(CFLAGS) $(LIB) -o $@ $^

build/%.o: src/%.c
	mkdir -p build
	$(CC) -c $(CFLAGS) $(INC) $< -o $@

build/%.o: src/%.cu
	mkdir -p build
	$(NVCC) $< -o $@ -lcuda -c $(INC) -I $(CUDA_INC) $(CUDA_FLAGS)

clean:
	rm -rf build
	rm -f csvconvert
