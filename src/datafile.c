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

float *dfm_getpoint(dps_t *X, int p) {
	//int offset = sizeof(int) + sizeof(uint32_t) + X->dim * p;

	//float *point = malloc(sizeof(float) * X->dim);

	//memcpy(point, &X->v[p], sizeof(float) * X->dim);

	return &X->v[p * X->dim];
}

// TODO: Implement
//float *dfm_getpoint_deep(dps_t *X, int p, dp_t o) {
//	//int offset = sizeof(int) + sizeof(uint32_t) + X->dim * p;
//
//	float *point = malloc(sizeof(float) * X->dim);
//
//	memcpy(point, &X->v[p], sizeof(float) * X->dim);
//
//	return point;
//}

void df_mmap(int fd, dps_t *X) {
	int allocsize = X->len * X->dim * sizeof(float);
	printf("Trying to alloc %d bytes\n", allocsize);

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
