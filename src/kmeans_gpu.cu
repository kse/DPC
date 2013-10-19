#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
/* include MTGP pre-computed parameter sets */
#include <curand_mtgp32dc_p_11213.h>

#include <float.h>
#include <stdbool.h>

// Borrowed from a curand example :)
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
		printf("Error at %s:%d\n",__FILE__,__LINE__); \
		exit(EXIT_FAILURE);}} while(0)

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
	    printf("Error at %s:%d\n",__FILE__,__LINE__); \
	    exit(EXIT_FAILURE);}} while(0)

extern "C" {
	#include <datafile.h>
	#include <kmeans_gpu.h>
	#include <kmeans_cpu_impl.h>
}

/*
 * Declarations of functions
 */
void gpu_init(cudaDeviceProp *prop);

__global__ void
gpu_rand_initiate(curandState_t *states, int seed);

__global__ void
kmeans_parallel_naive(curandState_t *state, float *X, float *C, int *out, 
		int dim, int Ccount, int Xlen, int blocks, float cost);

extern "C"
datapoint_array_t *
kmeans_parallel_gpu_init(dps_t *X, int k) {
	// Proceed like CPU.
	// Find a random initializer, begin uploading data to the GPU.
	// User helper functions to find out if we can fit all of our data on the
	// GPU.
	// Begin throwing copying data to the GPU ASAP, it takes time.
	
	/*
	 * Do stuff like checking if we have a device, and get the device
	 * properties.
	 */
	cudaDeviceProp prop;
	gpu_init(&prop);

	/*
	 * TODO:
	 * Eventually we want around 2K threads running. This should saturate most
	 * modern GPU's.
	 */
	dim3 numThreads(256,1,1);
	dim3 numBlocks(6,1,1);


	// Figure out if we can fit the data on the GPU
	size_t mem = prop.totalGlobalMem;

	if(mem < X->len * X->dim * sizeof(float)) {
		fprintf(stderr, "Not enough GPU memory to load dataset\n");
		exit(EXIT_FAILURE);
	}


	// The seed element used for k-means||
	int initial = rand() % X->len;
	int Ccount = 1;

	// Device pointers for X and C.
	float *d_X;
	float *d_C;
	int   *d_O;  // Storage for output.
		

	/*
	 * Create C storage, deeply (It has it's own array).
	 */
	datapoint_array_t *C;
	datapoint_array_new(&C, X->dim, 1);
	datapoint_array_add(C, &X->v[X->dim * initial]);

	CUDA_CALL( cudaMalloc(&d_X, X->len * X->dim * sizeof(float)) );

	// We allow a maximum presampling of 3*k, which should be a trivial number
	// of elements. Right now, d_Cs is a pointer into X.
	CUDA_CALL( cudaMalloc(&d_C, 3 * k * X->dim * sizeof(float)) );

	CUDA_CALL( cudaMalloc(&d_O, X->len * sizeof(int)) );
	CUDA_CALL( cudaMemset(d_O, '\0', X->len * sizeof(int)) );

	CUDA_CALL( cudaMemcpy(d_X, X->v, X->len * X->dim * sizeof(float), 
				cudaMemcpyHostToDevice) );

	CUDA_CALL( cudaMemcpy(d_C, C->v, X->dim *
				sizeof(float), cudaMemcpyHostToDevice) );


	/*
	 * Random state storage for kernels
	 */
	curandState_t *randState;

	CUDA_CALL( cudaMalloc(&randState, sizeof(curandState_t) * numThreads.x * 
				numBlocks.x) );

	gpu_rand_initiate<<<numBlocks, numThreads>>>(randState, time(NULL));


	float phiOfX;

	// Memory allocated for simple output. 
	// An int for each x ∈ X, a 1 after each run of kmeans_parallel_naive
	// means that it was added to C.
	int *O = (int *)malloc(sizeof(int) * X->len);

	/*
	 * BEGIN FUNKY LOOP
	 */
	while(Ccount < 3 * k) {
		phiOfX = cost(X, C);
		kmeans_parallel_naive<<<numBlocks, numThreads>>>(randState, 
				d_X, d_C, d_O, X->dim, Ccount,
				X->len, numBlocks.x, phiOfX);

		CUDA_CALL( cudaPeekAtLastError() );

		CUDA_CALL( cudaMemcpy(O, d_O, X->len * sizeof(int),
					cudaMemcpyDeviceToHost) );

		for(int i = 0; i < X->len; i++) {
			if(O[i] == 1) {
				datapoint_array_add(C, &X->v[X->dim * i]);

				if(C->len == 3 * k)
					break;
			}
		}

		if(Ccount == C->len) 
			printf("Made it through without adding an element :(\n");
		Ccount = C->len;
		CUDA_CALL( cudaMemset(d_O, '\0', X->len * sizeof(int)) );
		CUDA_CALL( cudaMemcpy(d_C, C->v, X->dim *
					sizeof(float), cudaMemcpyHostToDevice) );

		//break; // TODO: Remove
	}

	printf("Select %d centers\n", C->len);

	datapoint_array_t *Cprime;
	datapoint_array_new(&Cprime, X->dim, 0);

	kmeanspp_init(C, Cprime, k);

	datapoint_array_t *res = NULL;
	datapoint_array_deepcopy(&res, Cprime);
	datapoint_array_free(Cprime);
	datapoint_array_free(C);

	CUDA_CALL(cudaFree(randState));
	CUDA_CALL(cudaFree(d_O));
	CUDA_CALL(cudaFree(d_C));
	CUDA_CALL(cudaFree(d_X));

	return res;
}

__global__ void
gpu_rand_initiate(curandState_t *states, int seed) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	curand_init(seed, i, 0, &states[i]);
}

/*
 * Load C's into shared memory, load X's into shared iteratively, and write an
 * output vector containing the distance squared of all X's
 *
 * Prereq: Space in shared memory for C's + X
 */
__global__ void 
kmeans_parallel_calcdist(curandState *state, float *X, int *Cs, float *out, 
		int dim, int Ccount, int Xlen) {
	extern __shared__ float C[];
	//int x[4];

	//int i = threadIdx.x + blockDim.x * blockIdx.x;

	/*
	 * Start by loading all the C's into shared memory
	 */

	int j = threadIdx.x;
	// Each thread loads (a|several) float, don't do it if we're out of scope.
	while(j < Ccount * dim) {
		C[j] = X[Cs[j]];
		j += blockDim.x;
	}

	__syncthreads();
}

__device__ float
gpu_dist_naive(float *X, float *C, int dim, int Ccount) {
	float distsum = 0.0f;
	float minsum = FLT_MAX;

	for(int i = 0; i < Ccount; i++) {
		for(int j = 0; j < dim; j++) {
			float v =  X[j] - C[dim * i + j];
			distsum += powf((float)v, 2.0f);
		}

		if(distsum < minsum) {
			minsum = distsum;
		}

		distsum = 0.0f;
	}

	return minsum;
}

/*
 * Write a kernel naively reads C and x ∈ X, calculates the distance, and
 * updates an output vector with a 1 if the vector ends up in C.
 */
__global__ void
kmeans_parallel_naive(curandState_t *state, float *X, float *C, int *out, 
		int dim, int Ccount, int Xlen, int blocks, float cost) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	// Copy it to register for speed.
	curandState_t randState = state[i];

	// Distance from current x to group C, squared.
	float distsq;

	for(int j = i; j < Xlen; j += blockDim.x * blocks) {
		distsq = gpu_dist_naive(&X[j * dim], C, dim, Ccount);
		distsq = distsq / cost;

		distsq *= 50;

		if(curand(&state[i]) < UINT_MAX * distsq) {
			out[j] = 1;
		}
	}

	state[i] = randState;
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
