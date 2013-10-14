#include <stdint.h>
#include <unistd.h>

struct dps {
	int dim;
	int len;
	float *v;
};

struct dp {
	int dim;
	float *v;
};

typedef struct dps dps_t;
typedef struct dp dp_t;

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
