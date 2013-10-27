#define _GNU_SOURCE

#include "datafile.h"
#include "kmeans_cpu_impl.h"

#include <stdio.h>
#include <stdlib.h>

#include <errno.h>
#include <stdint.h>

#include <float.h>
#include <math.h>

// For memset
#include <string.h>

/*
 * Runs the k-means|| initialization to find our initial k centers.
 */
datapoint_array_t *
kmeans_parallel_init(dps_t *X, int k) {
	dp_t x;
	x.dim = X->dim;

	int initial = rand() % X->len;
	float *in = dfm_getpoint(X, initial);

	/*
	 * Create storage for our selected set, and add the initial point
	 */
	datapoint_array_t *C;
	datapoint_array_new(&C, X->dim, 0);
	datapoint_array_add(C, in);

	float phi = cost(X, C);
	float p = 0.0;

	datapoint_array_t *Cprime;
	datapoint_array_new(&Cprime, C->dim, 0);

	int logphi = (int)logf(phi);

	for(int j = 0; j < logphi; j++) {
		/*
		 * Break out of our loop early?
		 * TODO: Really?
		 */
		if(C->len > 3*k - 1) {
			break;
		}

		for(int i = 0; i < X->len; i++) {
			x.v = dfm_getpoint(X, i);

			p = dist(&x, C);
			p = powf(p, 2);
			p *= k/2;

			p = p/phi;

			int r = rand();
			if(r < RAND_MAX * p) {
				datapoint_array_add(Cprime, x.v);
			}

			p = 0.0f;
		}

		if(Cprime->len > 0) {
			//printf("%f, ", phi);
			//printf("%d, ", j);
			datapoint_array_merge(C, Cprime);
			phi = cost(X, C);
		}
	}

	/*
	for(int i = 0; i < C->len; i++) {
		for(int j = 0; j < C->dim; j++) {
			if(j != C->dim - 1) {
				printf("%f,", C->v[i][j]);
			} else {
				printf("%f\n", C->v[i][j]);
			}
		}
	}
	*/

	//kmeanspp_init(C, Cprime, k);

	//printf(" %f\n", phi);
	//printf("%d datapoints\n", C->len);
	//printf("%f cost of C unminimized\n", cost(X, C));
	//printf("%f cost of cprime\n", cost(X, Cprime));
	
	//printf("Done sampling\n");
	
	//datapoint_array_t *res = NULL;
	//datapoint_array_deepcopy(&res, Cprime);
	datapoint_array_free(Cprime);

	return C;
}

datapoint_array_t *reduce_centers(datapoint_array_t *C, int k) {
	datapoint_array_t *Cprime;
	datapoint_array_new(&Cprime, C->dim, 0);

	kmeanspp_init(C, Cprime, k);

	datapoint_array_t *res = NULL;
	datapoint_array_deepcopy(&res, Cprime);

	datapoint_array_free(Cprime);
	datapoint_array_free(C);

	return res;
}

void kmeanspp_impl(dps_t *X, datapoint_array_t *C) {
	int i, j, k;
	float cos;
	//float prev_cos;

	datapoint_t c;
	datapoint_t x;
	c.dim = C->dim;
	x.dim = C->dim;

	float min;
	int   minc = 0;
	float tsum;

	if(!C->deep) {
		fprintf(stderr, "Argument C to kmeanspp_impl must be deep copy\n");
		exit(EXIT_FAILURE);
	}

	// Our buckets
	int count[C->len];
	float *p = malloc(C->dim * C->len * sizeof(float));

	if(p == NULL) {
		fprintf(stderr, "Error mallocing temporary storage: %s\n",
				strerror(errno));
		exit(EXIT_FAILURE);
	}

	cos = cost(X, C);

	//printf("Initial cost %f\n", cos);

	int its = 0;

	do {
		// Remember previous total cost.
		//prev_cos = cos;

		// Reset counters over each iteration.
		memset(count, '\0', C->len * sizeof(int));
		memset(p,     '\0', C->dim * C->len * sizeof(float));

		// Loop over all points stored in X.
		for(j = 0; j < X->len; j++) {
			min = FLT_MAX;
			x.v = dfm_getpoint(X, j);

			for(int i = 0; i < C->len; i++) {
				tsum = 0.0f;
				c.v = C->v[i];

				for(k = 0; k < X->dim; k++) {
					tsum += powf(x.v[k] - c.v[k], 2);
				}

				tsum = sqrtf(tsum);

				if(tsum < min) {
					minc = i;
					min = tsum;
				}
			}
			
			for(i = 0; i < X->dim; i++) {
				//printf("p[%d * %d + %d] = %f\n", minc, C->dim, i, p[minc * C->dim + i]);
				p[minc * C->dim + i] += x.v[i];
				count[minc]++;
			}
		}

		/*
		for(i = 0; i < C->len; i++) {
			printf("%d ", count[i]);
		}
		printf("\n");
		*/

		for(i = 0; i < C->len; i++) {
			for(k = 0; k < C->dim; k++) {
				C->v[i][k] = 2 * p[C->dim * i + k]/count[i];
			}
		}

		cos = cost(X, C);
		//printf("Prev_Cost is:%f\n", prev_cos);
		//printf("Cost is:     %f\n", cos);

		its ++;
	} while(its < 50);

	printf("Final cost is:     %f\n", cos);

	free(p);
}

/*
 * k-means++ initialization. Used in the last step of k-means|| initialization.
 */
void kmeanspp_init(datapoint_array_t *X, datapoint_array_t *C, int k) {
	dp_t x;
	x.dim = X->dim;

	int init = rand() % X->len;

	datapoint_array_add(C, X->v[init]);

	while(C->len < k) {
		float cost =  costpp(X, C);
		for(int i = 0; i < X->len; i++) {
			x.v = X->v[i];

			float p = powf(dist(&x, C), 2);
			p      /= cost;

			int r = rand();

			if(r < RAND_MAX * p) {
				datapoint_array_add(C, x.v);
			}

			// Break out of loop early, if we've chosen a k'th point.
			if(C->len == k) {
				break;
			}
		}
	}
}

/*
 * Calculates in d dimensions from x to all elements in C,
 * and return the minumum
 */
float dist(dp_t *x, datapoint_array_t *C) {
	float min = FLT_MAX;
	float tsum = 0.0f;

	datapoint_t c;
	c.dim = C->dim;

	for(int i = 0; i < C->len; i++) {
		c.v = C->v[i];

		for(int j = 0; j < x->dim; j++) {
			tsum += powf(x->v[j] - c.v[j], 2);
		}

		tsum = sqrtf(tsum);

		if(tsum < min) {
			min = tsum;
		}

		tsum = 0.0f;
	}

	return min;
}

/*
 * Cost function taking different parameters than the other.
 */
float costpp(datapoint_array_t *X, datapoint_array_t *C) {
	dp_t x;
	x.dim = X->dim;

	float sum = 0.0f;

	for(int i = 0; i < X->len; i++) {
		x.v = X->v[i];

		sum += powf(dist(&x, C), 2);
	}

	return sum;
}

float cost(dps_t *X, datapoint_array_t *C) {
	dp_t x;
	x.dim = X->dim;

	float sum = 0.0f;

	for(int i = 0; i < X->len; i++) {
		x.v = dfm_getpoint(X, i);

		sum += powf(dist(&x, C), 2);
	}

	return sum;
}
