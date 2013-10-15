#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>

#include <datafile.h>

// For stat
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

// For sterror and errno
#include <errno.h>
#include <string.h>

// For mmap
#include <sys/mman.h>

// For open
#include <fcntl.h>

// For hcreate/search/destroy
#include <search.h>

// For specific integer types
#include <stdint.h>

#define uint unsigned int

/*
 * Purpose:
 * Read a CSV file containing data, and write out a binary representation of
 * the same file. The purpose of this is to minimize the amount of data we
 * actually have to read.
 * Data dimensions are seperated by commas, datapoints by newlines.
 * The output datafile contains the data, prefixed by a 32 bit integer, 
 * specifying the dimensions of the data.
 *
 * All datapoints are written as single precision floats.
 *
 * TODO
 * * Allow specifying output file, right now it's just 'output'.
 * * Make flag to specify if there is a header in the CSV.
 */

int read_point(char *mapped, uint length, int dims, float **ret,
		uint *input_pointer);

int main(int argc, char **argv) {
	if(argc < 2) {
		fprintf(stderr, "Please include a datafile to convert\n");
		exit(EXIT_FAILURE);
	}

	char *input = argv[1];
	struct stat in_stat;

	// Get file information
	if(stat(input, &in_stat) < 0) {
		fprintf(stderr, "Unable to open input: %s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}

	// Check if it is a regular file
	if(!S_ISREG(in_stat.st_mode)) {
		fprintf(stderr, "Not a regular file\n");
		exit(EXIT_FAILURE);
	}

	int in_fh;
	if((in_fh = open(input, 0)) < 0) {
		fprintf(stderr, "Unable to open input: %s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}

	char *in = (char *)mmap(NULL, in_stat.st_size, PROT_READ, MAP_PRIVATE,
			in_fh, 0);

	if(in == (void *)-1) {
		fprintf(stderr, "Unable to MMAP input file: %s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}

	if(hcreate(1000) < 0) {
		fprintf(stderr, "Unable to initiate hashmap: %s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}
	

	float *point = NULL;
	uint in_p = 0;
	int read = 0;
	int dim = 0;
	uint32_t length = 0;

	FILE *output = fopen("output", "w");

	while(in_p < in_stat.st_size) {
		read = read_point(in,
				in_stat.st_size,
				dim,
				&point,
				&in_p);

		// First iteration
		if(dim == 0) {
			dim = read;
			fwrite(&dim, sizeof(int), 1, output);
			fwrite(&length, sizeof(uint32_t), 1, output);

			char zero = 0;
			int leftover = sysconf(_SC_PAGESIZE) - 
				((sizeof(uint32_t) + sizeof(int)) % sysconf(_SC_PAGESIZE));

			//printf("Size is %lu\n", sizeof(uint32_t) + sizeof(int));
			//printf("Pagesize %ld\n", sysconf(_SC_PAGESIZE));
			//printf("Writing an additional %d bytes\n", leftover);

			for(int i = 0; i < leftover; i++) {
				fwrite(&zero, 1, 1, output);
			}
		}

		if(read != dim) {
			fprintf(stderr, "WARNING: Going from dim %d to %d\n", dim, read);
		}

		length++;

		fwrite(point, sizeof(float), dim, output);
	}

	// Update the data with the total length.
	fseek(output, (long)sizeof(int), SEEK_SET);
	fwrite(&length, sizeof(uint32_t), 1, output);

	fclose(output);
	free(point);

	munmap(in, in_stat.st_size);

	return EXIT_SUCCESS;
}

/*
 * Read a datapoint of (unknown) dimension d, returns a float array of size d.
 */
int read_point(char *mapped, uint length, int dims, float **ret,
		uint *input_pointer) {
	uint ret_size = dims;
	// If dims == 0, we still don't know the dimension of our datapoints.
	// So allocate an array of a certain size, then we have to handle 
	// expansions ourselves.
	if(dims == 0) {
		ret_size = 42;
	}

	if(*ret == NULL ) {
		*ret = malloc(sizeof(float) * ret_size);
		if(*ret == NULL) {
			fprintf(stderr, "Unable to allocate memory in (read_point): %s\n",
					strerror(errno));
			exit(EXIT_FAILURE);
		}
	}


	// Buffer to read data into.
	char buf[1024];

	// So we know to stop when we reach EOF.
	int at_eof = 0;

	// Telltale so we know if we've read a newline.
	int at_newline = 0;

	int dim = 0;

	while(!at_eof && !at_newline) {
		uint p = 0;
		ENTRY e, *ep;

		// Read a dimension, until we reach , or \n.
		while(1) {
			char m = mapped[*input_pointer + p];

			if(p > 1024 - 1) {
				fprintf(stderr, "Error, input to gettok too long\n");
				exit(EXIT_FAILURE);
			}

			buf[p] = m;
			p++;

			if(m == ',') {
				dim++;
				break;
			}

			if(m == '\n') {
				dim++;
				at_newline = 1;
				break;
			}

			if(*input_pointer + p >= length) {
				dim++;
				at_eof = 1;
				break;
			}
		}

		buf[p - 1] = '\0';
		*input_pointer += p;

		//printf("Dim is: '%d'\n", dim);

		char *delta;
		float v = strtof(buf, &delta);

		// We found something not a string.
		if(delta == buf) {
			//printf("Buf dim %d contains: '%s'\n",dim, buf);
			v = 0.0f;
			char sbuf[1024];
			sbuf[0] = '\0';
			sprintf(sbuf, "%d-", dim);
			strcat(sbuf, buf);

			// We are looking for buf;
			e.key = sbuf;
			ep = hsearch(e, FIND);

			// New element we haven't seen before
			if(ep == NULL) {
				sbuf[0] = '\0';
				sprintf(sbuf, "-%d", dim);
				intptr_t data = 0;

				// Here for clarity, the pointer is already correct.
				e.key = sbuf;

				ep = hsearch(e, FIND);
				if(ep == NULL) {
					// Okay, no data has been entered for this dimension,
					// start with 0.
					e.key = strdup(sbuf);
					e.data = (void *)0;

					if(e.key == NULL) {
						fprintf(stderr, "Unable to allocate hashkey: %s\n",
								strerror(errno));
						exit(EXIT_FAILURE);
					}

					ep = hsearch(e, ENTER);
					if(ep == NULL) {
						fprintf(stderr, "Hash table is full: %s\n",
								strerror(errno));
						exit(EXIT_FAILURE);
					}
				} else {
					data = (intptr_t)(ep->data) + 1;
					ep->data = (void *)data;
					v = (float)data;
				}

				sbuf[0] = '\0';
				sprintf(sbuf, "%d-", dim);
				strcat(sbuf, buf);

				e.key = strdup(sbuf);
				e.data = (void *)data;

				if(e.key == NULL) {
					fprintf(stderr, "Unable to allocate hashkey2: %s\n",
							strerror(errno));
					exit(EXIT_FAILURE);
				}

				ep = hsearch(e, ENTER);
				if(ep == NULL) {
					fprintf(stderr, "Hash table is full: %s\n",
							strerror(errno));
					exit(EXIT_FAILURE);
				}
			} else {
				int k = (intptr_t)ep->data;
				v = (float)k;
			}
		}

		// Check if ret is big enough, else realloc
		if(ret_size < (uint)dim) {
			printf("Realloccing to %d\n", ret_size);
			ret_size = 2 * ret_size;
			*ret = (float *)realloc((void *)*ret, ret_size * sizeof(float));
		}

		(*ret)[dim - 1] = v;
	}

	if((uint)dim < ret_size) {
		printf("Dealloccing to %d\n", dim);
		*ret = realloc((void *)*ret, dim * sizeof(float));
	}

	return dim;
}
