#ifndef DATAFILE_H
#define DATAFILE_H

#include <stdint.h>
#include <unistd.h>

#define INIT_DATAPOINTS_ARRAY_SIZE 4

/*
 * Datapoints structure. Laid out in continuous memory
 */
struct datapoints {
	int dim;      // Length of a single element in dim
	int len;      // # of elements of size dim in v
	float *v;
} __attribute__ ((aligned));

/*
 * A single datapoint
 */
struct datapoint {
	int dim;  // Length of v
	float *v;
} __attribute__ ((aligned));

/*
 * An array of datapoints, that don't point to continuous memory
 */
struct datapoint_array {
	int size;    // Allocated space for v
	int len;     // Length of v.
	int dim;     // Length of *v
	int deep;    // Whether or not each element in v is allocated for
				 // this struct
	float **v;   // The values.
} __attribute__ ((aligned));

typedef struct datapoints dps_t;
typedef struct datapoint dp_t;
typedef struct datapoint datapoint_t;
typedef struct datapoint_array datapoint_array_t;

/*
 * Open a datafile, exits on failure. Returns filehandle.
 */
int df_open(const char *path);

/*
 * Returns x, the number of datapoints in the file
 */
uint32_t df_length(int fd);

/*
 * Returns d, the dimension of each datapoint
 */
uint32_t df_dim(int fd);

/*
 * Return malloc'ed array containing one datapoint.
 * Retsize is set to the length of the array.
 */
float *dfm_getpoint(dps_t *X, int p);

/*
 * Memory map functions
 */
void df_mmap(int fd, dps_t *X);
void df_munmap(dps_t *X);

/*
 * Operations on the datapoint_array type.
 */
void datapoint_array_add(datapoint_array_t *A, float *p);
void datapoint_array_new(datapoint_array_t **A, int dim, int deep);
void datapoint_array_free(datapoint_array_t *A);
void datapoint_array_merge(datapoint_array_t *A, datapoint_array_t *B);
void datapoint_array_deepcopy(datapoint_array_t **dest, 
		datapoint_array_t *src);

#endif
