#ifndef KMEANS_GPU_H
#define KMEANS_GPU_H

datapoint_array_t * kmeans_parallel_gpu_init_naive(dps_t *X, int k);
datapoint_array_t * kmeans_parallel_gpu_init_v1(dps_t *X, int k);

#endif
