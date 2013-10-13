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
 * * Allow specifying output file
 * * Make flag to acknowledge header
 */

int main(int argc, char **argv) {
	if(argc < 2) {
		fprintf(stderr, "Please include a datafile to convert\n");
		exit(1);
	}

	char *input = argv[1];
	struct stat in_stat;

	// Get file information
	if(stat(input, &in_stat) < 0) {
		fprintf(stderr, "Unable to open input: %s\n", strerror(errno));
		exit(1);
	}

	// Check if it is a regular file
	if(!S_ISREG(in_stat.st_mode)) {
		fprintf(stderr, "Not a regular file\n");
		exit(1);
	}

	int in_fh;
	if((in_fh = open(input, 0)) < 0) {
		fprintf(stderr, "Unable to open input: %s\n", strerror(errno));
		exit(1);
	}

	char *in = (char *)mmap(NULL, in_stat.st_size, PROT_READ, MAP_PRIVATE, in_fh, 0);
	if(in == (void *)-1) {
		fprintf(stderr, "Unable to MMAP input: %s\n", strerror(errno));
		exit(1);
	}

	char buf[1024];
	int in_pointer = 0;
	int at_eof = 0;
	int dim = 0;
	int at_newline = 0;

	if(hcreate(1000) < 0) {
		fprintf(stderr, "Unable to initiate hashmap: %s\n", strerror(errno));
		exit(1);
	}

	while(!at_eof) {
		int p = 0;

		while(1) {
			char m = in[in_pointer + p];

			if(p > 1024 - 1) {
				fprintf(stderr, "Error, input to gettok too long\n");
				exit(1);
			}

			buf[p] = m;
			p++;

			if(m == ',') {
				dim++;
				break;
			}

			if(m == '\n') {
				at_newline = 1;
				break;
			}

			if(in_pointer + p >= in_stat.st_size) {
				at_eof = 1;
				break;
			}
		}

		buf[p - 1] = '\0';

		printf("Buf contains: '%s'\n", buf);

		char *delta;
		float v = strtof(buf, &delta);
		in_pointer += p;

		if(delta == buf) {
			printf("Dim is %d\n", dim);
		} else {
			printf("%f\n", v);
		}

		if(at_newline) {
			dim = 0;
			at_newline = 0;
		}
	}

	munmap(in, in_stat.st_size);

	return 0;
}
