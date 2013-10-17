#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <datafile.h>

extern "C" {
	#include <kmeans_gpu.h>
}

/*
 * Declarations of functions
 */
void gpu_init(cudaDeviceProp *prop);

extern "C"
datapoint_array_t *
kmeans_parallel_gpu_init(dps_t *X, int k) {
	// Proceed like CPU.
	// Find a random initializer, begin uploading data to the GPU.
	// User helper functions to find out if we can fit all of our data on the
	// GPU.
	// Begin throwing copying data to the GPU ASAP, it takes time.
	
	/*
	 * Do stuff like checking if we have a device.
	 */
	cudaDeviceProp prop;
	gpu_init(&prop);

	// Figure out if we can fit the data on the GPU
	size_t mem = prop.totalGlobalMem;

	if(mem < X->len * X->dim * sizeof(float)) {
		fprintf(stderr, "Not enough GPU memory to load dataset\n");
		exit(EXIT_FAILURE);
	}

	float *d_X;

	cudaError_t err = cudaMalloc(&d_X, X->len * X->dim * sizeof(float));
	if(err != cudaSuccess) {
		fprintf(stderr, "Error allocation X on GPU: %s\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	cudaFree(d_X);
	return NULL;
}

void gpu_init(cudaDeviceProp *prop) {
	int dev_count = 0;

	cudaError_t err = cudaGetDeviceCount(&dev_count);

	if(err != cudaSuccess) {
		printf("cudaGetDeviceCount returned %d\n-> %s\n", 
				(int)err, cudaGetErrorString(err));
		exit(EXIT_FAILURE);                                                     
	}

	if(dev_count == 0) {
		fprintf(stderr, "Detected 0 CUDA capable devices. Quitting\n");
		exit(EXIT_FAILURE);
	}

	cudaSetDevice(0);
	cudaGetDeviceProperties(prop, 0);
}
