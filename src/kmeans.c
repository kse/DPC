#define _GNU_SOURCE

#include "datafile.h"
#include "kmeans_cpu_impl.h"
#include "kmeans.h"

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>

#include <errno.h>
#include <stdint.h>

// For time()
#include <time.h>

#include <float.h>
#include <math.h>

/*
 * The amount k of clusters we want to find
 */
long int k = 4;

int main(int argc, char **argv) {
	// Sets global variables, exits on error.
	handle_options(argc, argv);

	// Seed our RNG
	srand(time(NULL));

	int datafd = df_open("output");

	// Run k-means||, to find our initial k's.
	init_kmeans(datafd);

	return 0;
}

void init_kmeans(int fd) {
	// Declare the main structure to keep track of our points.
	dps_t X;
	X.len = df_length(fd);
	X.dim = df_dim(fd);

	// MMAP our input file.
	df_mmap(fd, &X);

	/*
	 * Do the k-means|| initialization. That is, sample ψ = log(ϕ) times.
	 * TODO: Figure out how many rounds we want to run.
	 * Gives our sampled centers.
	 */
	datapoint_array_t *C = kmeans_parallel_init(&X, k);
	kmeanspp_impl(&X, C);

	//printf("Received %d initial centers\n", C->len);
	datapoint_array_free(C);
	df_munmap(&X);
}

void handle_options(int argc, char **argv) {
	int optret = 0;
	extern int   optind;
	extern char *optarg;

	static char opt[] = "k:";

	static struct option long_options[] = {
		{"clusters",  required_argument, 0, 'k'},
		{0,           0,                 0,  0 }     
	};
	
	while( (optret = getopt_long(argc, argv, opt, long_options, NULL)) != -1) {
		char *end = 0; // Store end of parsed. To check for errors

		switch(optret) {
			case 'k': 
				k = strtol(optarg, &end, 10);

				if(optarg == end || *end != '\0') {
					fprintf(stderr,
							"Invalid k option '%s', must be an integer\n",
							optarg);
					exit(EXIT_FAILURE);
				}

				if(errno == ERANGE) {
					// Uh-oh, over/under-flow
					fprintf(stderr,
							"Warning, k value overflow/underflow\n");
				}
				break;
			default: // We hit '?'
				//TODO: Print some usage information.
				exit(EXIT_FAILURE);
				break;
		}

	}
}
