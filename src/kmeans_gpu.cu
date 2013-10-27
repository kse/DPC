#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <curand_kernel.h>
#include <curand_mtgp32_host.h>

/* include MTGP pre-computed parameter sets */
#include <curand_mtgp32dc_p_11213.h>

#include <float.h>
#include <stdbool.h>

#include <errno.h>

// Borrowed from a curand example :)
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
		printf("Error at %s:%d\n",__FILE__,__LINE__); \
		exit(EXIT_FAILURE);}} while(0)

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
	    printf("Error at %s:%d\n",__FILE__,__LINE__); \
	    exit(EXIT_FAILURE);}} while(0)


#define XALLOC(p, t, x) do { p = (t)malloc(x); if(p == NULL) { \
	fprintf(stderr, "Malloc error %s-%d: %s\n", __FILE__, __LINE__, \
			strerror(errno)); \
	exit(EXIT_FAILURE); }\
	} while(0)

#define MIN(x, y) x < y ? x : y

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
kmeans_parallel_naive(curandStateMtgp32 *state, float *X, float *C, int *out, 
		int dim, int Ccount, int Xlen, int blocks, float cost, int k);

extern "C"
datapoint_array_t *
kmeans_parallel_gpu_init_naive(dps_t *X, int k) {
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
	dim3 numBlocks(1,1,1);


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

	curandStateMtgp32 *devMTGPStates;
	mtgp32_kernel_params *devKernelParams;

	/* Allocate space for prng states on device */
	CUDA_CALL(cudaMalloc((void **)&devMTGPStates, numBlocks.x * 
				sizeof(curandStateMtgp32)));

	/* Setup MTGP prng states */

	/* Allocate space for MTGP kernel parameters */
	CUDA_CALL(cudaMalloc((void**)&devKernelParams, sizeof(mtgp32_kernel_params)));

	/* Reformat from predefined parameter sets to kernel format, */
	/* and copy kernel parameters to device memory               */
	CURAND_CALL(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devKernelParams));

	/* Initialize one state per thread block */
	CURAND_CALL(curandMakeMTGP32KernelState(devMTGPStates, 
				mtgp32dc_params_fast_11213, devKernelParams, numBlocks.x, 1234));


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

	CUDA_CALL( cudaMemcpy(d_C, C->v[0], C->dim * sizeof(float),
				cudaMemcpyHostToDevice) );


	float phiOfX;

	// Memory allocated for simple output. 
	// An int for each x ∈ X, a 1 after each run of kmeans_parallel_naive
	// means that it was added to C.
	int *O = (int *)malloc(sizeof(int) * X->len);
	if(O == NULL) {
		fprintf(stderr, "Error allocating memory at %s.%d: %s\n",
				__FILE__, __LINE__, strerror(errno));
		exit(EXIT_FAILURE);
	}

	/*
	 * BEGIN FUNKY LOOP
	 */
	while(Ccount < 3 * k) {
		phiOfX = cost(X, C);  // The CPU loop for now.

		kmeans_parallel_naive<<<numBlocks, numThreads>>>(devMTGPStates, 
				d_X, d_C, d_O, X->dim, Ccount,
				X->len, numBlocks.x, phiOfX, k);

		CUDA_CALL( cudaPeekAtLastError() );

		//CUDA_CALL( cudaMemcpy(O, d_O, X->len * sizeof(int), cudaMemcpyDeviceToHost) );
		cudaError err = cudaMemcpy(O, d_O, X->len * sizeof(int),
				cudaMemcpyDeviceToHost);

		if(err != cudaSuccess) {
			fprintf(stderr, "O Copy error: %s\n",
					cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		for(int i = 0; i < X->len; i++) {
			if(O[i] == 1) {
				datapoint_array_add(C, &X->v[X->dim * i]);

				if(C->len == 3 * k)
					break;
			}
		}


		for(int i = Ccount; i < C->len; i++) {
			CUDA_CALL( cudaMemcpy(&d_C[C->dim * i], C->v[i], C->dim * sizeof(float),
						cudaMemcpyHostToDevice) );
		}

		Ccount = C->len;

		CUDA_CALL( cudaMemset(d_O, '\0', X->len * sizeof(int)) );
	}

	printf("Select %d centers\n", C->len);

	free(O);
	CUDA_CALL(cudaFree(d_O));
	CUDA_CALL(cudaFree(d_C));
	CUDA_CALL(cudaFree(d_X));

	return C;
}

__global__ void
gpu_rand_initiate(curandState_t *states, int seed) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	curand_init(seed, i + 24, 0, &states[i]);
}

__device__ float
gpu_dist_naive(float *X, float *C, int dim, int Ccount) {
	float distsum = 0.0f;
	float minsum = FLT_MAX;

	for(int i = 0; i < Ccount; i++) {
		for(int j = 0; j < dim; j++) {
			float v =  X[j] - C[dim * i + j];
			distsum += powf(v, 2.0f);
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
kmeans_parallel_naive(curandStateMtgp32 *state, float *X, float *C, int *out, 
		int dim, int Ccount, int Xlen, int blocks, float cost, int k) {

	int i = threadIdx.x + blockDim.x * blockIdx.x;

	// Copy it to register for speed.
	//curandState_t randState = state[blockIdx.x];

	// Distance from current x to group C, squared.
	float distsq;

	for(int j = i; j < Xlen; j += blockDim.x * blocks) {
		distsq = gpu_dist_naive(&X[dim * j], C, dim, Ccount);
		distsq *= k/2;
		distsq = distsq / cost;

		if(curand(&state[blockIdx.x]) < UINT_MAX * distsq) {
			out[j] = 1;
		}
	}

	//state[i] = randState;
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

__global__ void
kmeans_parallel_sample_v1(curandStateMtgp32 *state, int *out, float *dists,
		int Xs, int k, float cost) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	// Distance from current x to group C, squared.
	//float distsq;

	for(; i < Xs; i += blockDim.x * gridDim.x) {
		float distsq = dists[i]; // Contains the cost of the i'th vector.
		distsq *= k/2;
		distsq = distsq / cost;

		if(curand(&state[blockIdx.x]) < UINT_MAX * distsq) {
			out[i] = 1;
		}
	}
}

__global__ void
cost_kernel_v1(int dim, int Cs, float *C, int Xs, float *X, 
		float *sums, int *mins) {
	/* Iterator variable */
	int i, j, k;

	/* 
	 * Dynamic shared storage general pointer.
	 */
	extern __shared__ int shared[];

	/* Space for all of our centroids. */
	float *centroids = (float *) shared;

	/* Space for the results of our calculation */
	float *points    = (float *)&shared[dim * Cs];

	/*
	 * Our threadIdx.x runs from 1 to dim, so we do a simple check.
	 */
	for(j = threadIdx.x; j < dim * Cs; j += blockDim.x) {
		centroids[j] = C[j];
	}

	float min = FLT_MAX;
	int  minc = 0;
	float sum;

	for(k = 0; k < ((Xs-1) / blockDim.x * gridDim.x) + 1 ; k++) {
		int offset = gridDim.x * blockDim.x * dim * k + 
			blockDim.x * dim * blockIdx.x;

		__syncthreads();

		if(offset + threadIdx.x < Xs * dim) {
			for(j = threadIdx.x; j < dim * blockDim.x; j += blockDim.x) {
				points[j] = X[j + offset];
			}
		}

		__syncthreads();

		if(offset + threadIdx.x < Xs * dim) {
			min = FLT_MAX;
			for(i = 0; i < Cs; i++) {
				sum = 0.0f;

				for(j = 0; j < dim; j++) {
					sum += powf(points[threadIdx.x * dim + j]
							- centroids[i * dim + j], 2);
				}

				if(sum < min) {
					min = sum;
					minc = i;
				}
			}

			sums[gridDim.x * blockDim.x * k + blockDim.x * blockIdx.x + threadIdx.x] = min;
			mins[gridDim.x * blockDim.x * k + blockDim.x * blockIdx.x + threadIdx.x] = minc;
		}
	}
}

__host__ void mtgp32_init(curandStateMtgp32 **states,
		mtgp32_kernel_params **devKernelParams, int count) {

	/* Allocate space for prng states on device */
	CUDA_CALL(cudaMalloc((void **)states, count * 
				sizeof(curandStateMtgp32)));

	/* Setup MTGP prng states */

	/* Allocate space for MTGP kernel parameters */
	CUDA_CALL(cudaMalloc((void**)devKernelParams, sizeof(mtgp32_kernel_params)));

	/* Reformat from predefined parameter sets to kernel format, */
	/* and copy kernel parameters to device memory               */
	CURAND_CALL(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, *devKernelParams));

	/* Initialize one state per thread block */
	CURAND_CALL(curandMakeMTGP32KernelState(*states, 
				mtgp32dc_params_fast_11213, *devKernelParams, count, 1234));

}

/*
 * TODO: Decrease the thread count needed. We only need half.
 */
__global__ void
sum_reduction_kernel_v1(float *V, int length, float *out) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	//int offset;
	int j;

	extern __shared__ float T[];

	for(j = 0; j < (length - 1)/(gridDim.x * blockDim.x) + 1; j++) {
		/* Load our registers */
		if(j < length) {
			T[threadIdx.x] = V[i + blockIdx.x * blockDim.x +
				gridDim.x * blockDim.x * j];
		}

		__syncthreads();

		for(int stride = 1; stride < blockDim.x; stride *=2) {
			__syncthreads();

			if(j < length) {
				int ind = (threadIdx.x + 1) * stride * 2 - 1;
				if(ind < blockDim.x) {
					T[ind] += T[ind - stride];
				}
			}
		}

		__syncthreads();
		if(threadIdx.x == blockDim.x - 1) {
			out[blockIdx.x + gridDim.x * j] = T[blockDim.x - 1];
		}
	}
}

// TODO: Finish the loop inside.
// Also, does not work with several blocks :(
float do_reduce(float *d_minsum, int len, float *d_runsum) {
	// The length of the output data, used as offset.
	int runsum_len = 0;

	int f = ceilf(logf(len)/logf(1024.0f));

	// Len is originally not necesarrily a multiple of X->len, it is now!
	len = len + (1024 - (len%1024));

	// The first kernel looks at the original data, and start with offset 0.
	sum_reduction_kernel_v1<<<1, 1024, 1024 * sizeof(float)>>>(d_minsum,
			len, &d_runsum[runsum_len]);

	float sum[len/1024];

	CUDA_CALL( cudaMemcpy(sum, d_runsum, len/1024*sizeof(float),
				cudaMemcpyDeviceToHost) );

	float tsum = 0.0f;
	for(int i = 0; i < len/1024; i++) {
		//printf("d: %f\n", sum[i]);
		tsum += sum[i];
	}

	return tsum;


	/*
	for(int i = 2; i <= f; i++) {
		int oldrunsum = runsum_len;
		len = 1;

		//runsum_len += ceilf(len/powf(1024, i));
		int v = ceilf(len/ powf(1024, i));
		runsum_len += v + (1024 - (v%1024));

		//len = len + (1024 - (len%1024));

		printf("old Runsum inner: %d\n", oldrunsum);
		printf("Runsum inner: %d\n", runsum_len);

		sum_reduction_kernel_v1<<<1, 1024, 1024 * sizeof(float)>>>(
				&d_runsum[oldrunsum], len, &d_runsum[runsum_len]);
	}
	*/
}

template <int d>
__global__ void
cost_kernel_v2(int dim, int Cs, float *C, int Xs, float *X, 
		float *sums, int *mins);

extern "C"
datapoint_array_t *
kmeans_parallel_gpu_init_v1(dps_t *X, int k) {
	/*
	 * Do stuff like checking if we have a device, and get the device
	 * properties.
	 */
	cudaDeviceProp prop;
	gpu_init(&prop);

	/*
	 * TODO:
	 * Optimize these numbers for dimensions og cache-amount used per 
	 * algorithm.
	 */
	dim3 numThreads(1024,1,1);
	dim3 numBlocks(20,1,1);

	int max_centers = k * 3;

	// Figure out if we can fit the data on the GPU
	size_t mem = prop.totalGlobalMem;

	if(mem < X->len * X->dim * sizeof(float)) {
		fprintf(stderr, "Not enough GPU memory to load dataset\n");
		exit(EXIT_FAILURE);
	}


	// The seed element used for k-means||
	int initial = rand() % X->len;
	//initial = 0;
	int Ccount = 1;

	// Device pointers for X and C.
	float *d_X;
	float *d_C;
	int   *d_O;  // Storage for output.

	/*
	 * Stores the Cost() we compute on the GPU
	 */
	int   *d_min;    // Minimum c for each x
	float *d_minsum; // Minimum cost for each x

	curandStateMtgp32 *devMTGPStates;
	mtgp32_kernel_params *devKernelParams;

	mtgp32_init(&devMTGPStates, &devKernelParams, numBlocks.x);

	/*
	 * Create C storage, deeply (It has it's own array).
	 */
	datapoint_array_t *C;
	datapoint_array_new(&C, X->dim, 1);
	datapoint_array_add(C, &X->v[X->dim * initial]);

	CUDA_CALL( cudaMalloc(&d_X, X->len * X->dim * sizeof(float)) );

	// We allow a maximum presampling of 3*k, which should be a trivial number
	// of elements. Right now, d_Cs is a pointer into X.
	CUDA_CALL( cudaMalloc(&d_C, max_centers * X->dim * sizeof(float)) );

	/* Ready space for output of selected centers in k-means|| iteration */
	CUDA_CALL( cudaMalloc(&d_O, X->len * sizeof(int)) );
	CUDA_CALL( cudaMemset(d_O, '\0', X->len * sizeof(int)) );

	/* Copy data to GPU memory */
	CUDA_CALL( cudaMemcpy(d_X, X->v, X->len * X->dim * sizeof(float), 
				cudaMemcpyHostToDevice) );

	/* Copy the initial center to GPU memory */
	CUDA_CALL( cudaMemcpy(d_C, C->v[0], C->dim * sizeof(float),
				cudaMemcpyHostToDevice) );

	/* Storage for min c ∈ C d²(x, C) and d²(x, C) */
	CUDA_CALL( cudaMalloc(&d_min, X->len * sizeof(int)) );

	// We overallocate to fit a block-size of 1024, this is so we can compute
	// the sum easily on the GPU.
	int minsum_len = X->len + (1024 - (X->len%1024));
	//printf("Minsum_len: %d\n", minsum_len);
	CUDA_CALL( cudaMalloc(&d_minsum, minsum_len * sizeof(float)) );

	// TODO: Allow calculation of sums that top 1073741823 elements.
	// Right now, because of the fixed size, we can't.

	// Log(1024)
	int f = ceilf(logf(X->len)/logf(1024.0f));
	int runsum_len;// = ceilf(X->len/1024);

	for(int i = 1; i < f; i++) {
		int v = ceilf(X->len/ powf(1024, i));
		runsum_len += v + (1024 - (v%1024));
	}

	//printf("Runsum outer: %d\n", runsum_len);

	float *d_runsum;

	cudaError err = cudaMalloc(&d_runsum, runsum_len * sizeof(float));

	if(err != cudaSuccess) {
		fprintf(stderr, "Error alloccing runsum: %s\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Memory allocated for simple output. 
	// An int for each x ∈ X, a 1 after each run of kmeans_parallel_naive
	// means that it was added to C.
	int *O = (int *)malloc(sizeof(int) * X->len);
	if(O == NULL) {
		fprintf(stderr, "Error allocating memory at %s.%d: %s\n",
				__FILE__, __LINE__, strerror(errno));
		exit(EXIT_FAILURE);
	}

	while(Ccount < max_centers) {
		cudaError err;

		int shared_size = C->len * C->dim * sizeof(float)
			+ numThreads.x * sizeof(float) * X->dim;

		cost_kernel_v2<2><<<numBlocks, numThreads, shared_size>>>(X->dim, 
				C->len, d_C, X->len, d_X, d_minsum, d_min);

		cudaDeviceSynchronize(); // Nice for checking if there is an error.

		if(cudaPeekAtLastError() != cudaSuccess) {
			fprintf(stderr, "Error computing cost: %s\n",
					cudaGetErrorString(cudaPeekAtLastError()));
			exit(EXIT_FAILURE);
		};

		/* For some reason we need to set this before cost_kernel_v1.
		 * TODO: Investigate
		 */
		CUDA_CALL( cudaMemset(&d_minsum[X->len], '\0',
					(1024 - (X->len%1024)) * sizeof(float)) );

		// Clear previous usage.
		CUDA_CALL( cudaMemset(d_runsum, '\0', runsum_len * sizeof(float)) );

		// Compute ϕ_X(C)
		float phi = do_reduce(d_minsum, X->len, d_runsum);

		kmeans_parallel_sample_v1<<<numBlocks, numThreads>>>(devMTGPStates,
				d_O, d_minsum, X->len, k, phi);

		cudaDeviceSynchronize();

		CUDA_CALL( cudaPeekAtLastError() );

		err = cudaMemcpy(O, d_O, X->len * sizeof(int),
				cudaMemcpyDeviceToHost);

		if(err != cudaSuccess) {
			fprintf(stderr, "O Copy error: %s\n",
					cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}


		for(int i = 0; i < X->len; i++) {
			if(O[i] == 1) {
				datapoint_array_add(C, &X->v[X->dim * i]);

				if(C->len == max_centers)
					break;
			}
		}

		for(int i = Ccount; i < C->len; i++) {
			CUDA_CALL( cudaMemcpy(&d_C[C->dim * i],
						C->v[i], C->dim * sizeof(float),
						cudaMemcpyHostToDevice) );
		}

		Ccount = C->len;

		CUDA_CALL( cudaMemset(d_O, '\0', X->len * sizeof(int)) );
	}

	printf("Selected %d centers\n", C->len);

	free(O);
	CUDA_CALL( cudaFree(d_runsum) );
	CUDA_CALL(cudaFree(d_min));
	CUDA_CALL(cudaFree(d_minsum));
	CUDA_CALL(cudaFree(d_O));
	CUDA_CALL(cudaFree(d_C));
	CUDA_CALL(cudaFree(d_X));

	return C;
}


__host__ void
kmeans_gpu_v1(float *d_X, float *d_C, int Xs, int Cs, float *sums, int *mins) {

}

template <int d>
__global__ void
cost_kernel_v2(int dim, int Cs, float *C, int Xs, float *X, 
		float *sums, int *mins) {
	/* Iterator variable */
	int i, j, k;
	int regstor[d];

	/* 
	 * Dynamic shared storage general pointer.
	 */
	extern __shared__ int shared[];

	/* Space for all of our centroids. */
	float *centroids = (float *) shared;

	/* Space for the results of our calculation */
	float *points    = (float *)&shared[dim * Cs];

	/*
	 * Our threadIdx.x runs from 1 to dim, so we do a simple check.
	 */
	for(j = threadIdx.x; j < dim * Cs; j += blockDim.x) {
		centroids[j] = C[j];
	}

	float min = FLT_MAX;
	int  minc = 0;
	float sum;

	for(k = 0; k < ((Xs-1) / blockDim.x * gridDim.x) + 1 ; k++) {
		int offset = gridDim.x * blockDim.x * dim * k + 
			blockDim.x * dim * blockIdx.x;

		__syncthreads();

		if(offset + threadIdx.x < Xs * dim) {
			for(j = threadIdx.x; j < dim * blockDim.x; j += blockDim.x) {
				points[j] = X[j + offset];
			}
		}

		__syncthreads();

		if(offset + threadIdx.x < Xs * dim) {
			min = FLT_MAX;
			for(i = 0; i < Cs; i++) {
				sum = 0.0f;

				for(j = 0; j < dim; j++) {
					sum += powf(points[threadIdx.x * dim + j]
							- centroids[i * dim + j], 2);
				}

				if(sum < min) {
					min = sum;
					minc = i;
				}
			}

			sums[gridDim.x * blockDim.x * k + blockDim.x * blockIdx.x + threadIdx.x] = min;
			mins[gridDim.x * blockDim.x * k + blockDim.x * blockIdx.x + threadIdx.x] = minc;
		}
	}
}
