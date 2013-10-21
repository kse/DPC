#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>

#include "datafile.h"

// For stat
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// For sterror and errno
#include <errno.h>
#include <string.h>

// For specific integer types
#include <stdint.h>

// For mmap
#include <sys/mman.h>

int df_open(const char *path) {
	int f = open(path, O_NOATIME | O_RDONLY);

	if(f == -1) {
		fprintf(stderr, "Unable to open datafile: %s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}

	return f;
}

uint32_t df_dim(int fd) {
	int val = 0;
	off_t inpos = lseek(fd, 0, SEEK_CUR);

	if(lseek(fd, 0, SEEK_SET) == -1) {
		fprintf(stderr, "Unable to seek in datafile: %s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}

	if(read(fd, &val, sizeof(int)) == -1) {
		fprintf(stderr, "Short read of length from datafile: %s\n",
				strerror(errno));
		exit(EXIT_FAILURE);
	}

	lseek(fd, inpos, SEEK_SET);

	return val;
}

uint32_t df_length(int fd) {
	uint32_t val = 0;
	off_t inpos = lseek(fd, 0, SEEK_CUR);

	if(lseek(fd, (off_t)sizeof(int), SEEK_SET) == -1) {
		fprintf(stderr, "Unable to seek in datafile: %s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}

	if(read(fd, &val, sizeof(uint32_t)) == -1) {
		fprintf(stderr, "Short read of length from datafile: %s\n",
				strerror(errno));
		exit(EXIT_FAILURE);
	}

	lseek(fd, inpos, SEEK_SET);

	return val;
}

inline float *dfm_getpoint(dps_t *X, int p) {
	return &X->v[p * X->dim];
}

void df_mmap(int fd, dps_t *X) {
	int allocsize = X->len * X->dim * sizeof(float);
	//printf("Trying to alloc %d bytes\n", allocsize);

	float *in = (float *)mmap(NULL,
			allocsize,
			PROT_READ, MAP_PRIVATE, fd,
			sysconf(_SC_PAGESIZE));

	if(in == (void *)-1) {
		fprintf(stderr, "Unable to MMAP input file: %s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}

	X->v = in;
}

void df_munmap(dps_t *X) {
	munmap(X->v, X->len * X->dim * sizeof(float));
}

void datapoint_array_new(datapoint_array_t **A, int dim, int deep) {
	*A = malloc(sizeof(struct datapoint_array));

	if(*A == NULL) {
		fprintf(stderr, "Cannot allocate new dpa: %s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}

	(*A)->size = 4;
	(*A)->len  = 0;
	(*A)->dim  = dim;
	(*A)->deep = deep ? 1 : 0;

	(*A)->v = malloc(INIT_DATAPOINTS_ARRAY_SIZE * sizeof(float *));

	if((*A)->v == NULL) {
		fprintf(stderr, "Cannot allocate new dpa store: %s\n",
				strerror(errno));
		exit(EXIT_FAILURE);
	}
}

void datapoint_array_add(datapoint_array_t *A, float *p) {
	// NB: Make damned sure the point is of the same dimension as A->dim
	
	if(A->size == A->len) {
		// Expand to double size be realloccing.
		float **new = realloc(A->v, A->size * 2 * sizeof(float *));
		if(new == NULL) {
			fprintf(stderr, "Unable to realloc dpa_t: %s\n", strerror(errno));
			exit(EXIT_FAILURE);
		}

		A->v = new;
		A->size *= 2;
	}

	if(A->deep == 1) {
		A->v[A->len] = malloc(A->dim * sizeof(float));
		memcpy(A->v[A->len], p, A->dim * sizeof(float));
	} else {
		A->v[A->len] = p;
	}
	A->len++;
}

void datapoint_array_free(datapoint_array_t *A) {
	if(A->deep == 1) {
		for(int i = 0; i < A->len; i++) {
			free(A->v[i]);
		}
	}

	free(A->v);
	free(A);
}

void datapoint_array_merge(datapoint_array_t *A, datapoint_array_t *B) {
	// TODO: Implement handling of deep datapoint arrays.
	if(B->len == 0) {
		return;
	}

	if((A->size - A->len) < B->len ) {
		float **new = realloc(A->v, (A->size * 2 + B->len) * sizeof(float *));

		if(new == NULL) {
			fprintf(stderr, "Unable to merge dpa_t: %s\n", strerror(errno));
			exit(EXIT_FAILURE);
		}

		A->v = new;
		A->size *= 2;
		A->size += B->len;
	}

	memcpy(&(A->v[A->len]), B->v, B->len * sizeof(float *));
	A->len += B->len;
	B->len = 0;
}

void datapoint_array_deepcopy(datapoint_array_t **dest,
		datapoint_array_t *src) {
	*dest = malloc(sizeof(datapoint_array_t));

	(*dest)->len  = src->len;
	(*dest)->dim  = src->dim;
	(*dest)->deep = 1;

	(*dest)->v = malloc(src->len * sizeof(float *));

	memcpy((*dest)->v, src->v, src->len * sizeof(float *));

	for(int i = 0; i < src->len; i++) {
		(*dest)->v[i] = malloc(sizeof(float) * src->dim);
		memcpy((*dest)->v[i], src->v[i], sizeof(float) * src->dim);
	}
}
