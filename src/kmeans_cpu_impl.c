#define _GNU_SOURCE

#include "datafile.h"
#include "kmeans_cpu_impl.h"

#include <stdio.h>
#include <stdlib.h>

#include <errno.h>
#include <stdint.h>

#include <float.h>
#include <math.h>

void sample(dps_t *X, datapoint_array_t *C, int k) {
	dp_t x;
	x.dim = X->dim;

	float phi = cost(X, C);
	float p = 0.0;
	printf("Initial cost is %f\n", phi);

	datapoint_array_t *Cprime;
	datapoint_array_new(&Cprime, C->dim);

	int logphi = (int)logf(phi);
	printf("Logphi: %d\n", logphi);


	for(int j = 0; j < logphi; j++) {
		for(int i = 0; i < X->len; i++) {
			x.v = dfm_getpoint(X, i);

			p = dist(&x, C);
			p = powf(p, 2);
			p *= k/2;

			float pre_p = p;
			p = p/phi;

			int r = rand();
			if(r < RAND_MAX * p) {
				printf("Selected %d\n", i);
				printf("Prob is: %.20f\n", p);
				printf("Pre-P is %f\n", pre_p);

				datapoint_array_add(Cprime, x.v);
			}

			p = 0.0f;
		}

		if(Cprime->len > 0) {
			datapoint_array_merge(C, Cprime);
			phi = cost(X, C);
			printf("Cost is %f\n", phi);
		}
	}

	printf("Selected %d datapoints\n", C->len);

	datapoint_array_free(Cprime);
}

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
