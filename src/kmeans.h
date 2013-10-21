#ifndef KMEANS_H

/*
 * Handle all input options, don't clutter main() too much.
 */
void handle_options(int argc, char **argv);
void write_datapoint_array(datapoint_array_t *C, char *of);

void init_kmeans(int fd);

struct kmean_impl {
	datapoint_array_t *(*init)(dps_t *X, int k);
};

#endif
