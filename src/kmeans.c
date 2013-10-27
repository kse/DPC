#define _GNU_SOURCE

#include "datafile.h"
#include "kmeans_cpu_impl.h"
#include "kmeans_gpu.h"
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
int on_cpu = 0;
int dumb   = 0;
int blocksize = 1024;

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

	struct timespec start, end;


	if(on_cpu) {
		/*
		 * Do the k-means|| initialization. That is, sample ψ = log(ϕ) times.
		 * TODO: Figure out how many rounds we want to run.
		 * Gives our sampled centers.
		 */

		clock_gettime(CLOCK_REALTIME, &start);
		datapoint_array_t *C = kmeans_parallel_init(&X, k);
		clock_gettime(CLOCK_REALTIME, &end);

		//write_datapoint_array(C, "preoutput.csv");

		//C = reduce_centers(C, k);

		//write_datapoint_array(C, "reducedoutput.csv");

		/*
		 * Run the k-means Lloyd iteration, this is the time waster.
		 * So maybe we should do something with the data afterwards?
		 * Like print it, so we can plot it..
		 */
		//kmeanspp_impl(&X, C);

		//write_datapoint_array(C, "output.csv");

		/*
		 * Let go of our data.
		 */
		datapoint_array_free(C);

	} else if(dumb) {
		clock_gettime(CLOCK_REALTIME, &start);
		datapoint_array_t *C = kmeans_parallel_gpu_init_naive(&X, k);
		clock_gettime(CLOCK_REALTIME, &end);

		//write_datapoint_array(C, "preoutput.csv");

		//C = reduce_centers(C, k);

		//write_datapoint_array(C, "reducedoutput.csv");

		//kmeanspp_impl(&X, C);

		//write_datapoint_array(C, "output.csv");

		datapoint_array_free(C);

	} else {
		clock_gettime(CLOCK_REALTIME, &start);
		datapoint_array_t *C = kmeans_parallel_gpu_init_v1(&X, k);
		clock_gettime(CLOCK_REALTIME, &end);

		//write_datapoint_array(C, "preoutput.csv");

		//C = reduce_centers(C, k);

		//write_datapoint_array(C, "reducedoutput.csv");

		//kmeanspp_impl(&X, C);

		//write_datapoint_array(C, "output.csv");

		datapoint_array_free(C);
	}

	long long int nsec = (end.tv_sec - start.tv_sec) * 1000000000 
		+ (end.tv_nsec - start.tv_nsec);

	printf("k-means|| time: %llds.%lldus\n", nsec/1000000000,
			(nsec%1000000000)/1000);

	df_munmap(&X);
}

void write_datapoint_array(datapoint_array_t *C, char *of) {
	FILE *file = fopen(of, "w");
	if(file == NULL) {
		fprintf(stderr, "Unable to write datapoints: %s\n",
				strerror(errno));
		exit(EXIT_FAILURE);
	}

	for(int i = 0; i < C->len; i++) {
		for(int j = 0; j < C->dim; j++) {
			fprintf(file, "%f", C->v[i][j]);

			if(j != C->dim - 1) {
				fprintf(file, ", ");
			}
		}

		fprintf(file, "\n");
	}

	fclose(file);
}

void handle_options(int argc, char **argv) {
	int optret = 0;
	extern int   optind;
	extern char *optarg;

	static char opt[] = "k:cdb:";

	static struct option long_options[] = {
		{"clusters",  required_argument, 0, 'k'},
		{"cpu",       no_argument,       0, 'c'},
		{"dumb",      no_argument,       0, 'd'},
		{"blocksize", required_argument, 0, 'b'},
		{0,           0,                 0,  0 }     
	};
	
	while( (optret = getopt_long(argc, argv, opt, long_options, NULL)) != -1) {
		char *end = 0; // Store end of parsed. To check for errors

		// TODO:
		// * Accept input file
		// * Take a flag that tells us to use the GPU or not.

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
			case 'c':
				on_cpu = 1;
				break;
			case 'd':
				dumb = 1;
				break;
			case 'b': 
				blocksize = strtol(optarg, &end, 10);

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
