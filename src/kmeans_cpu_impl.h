float cost(dps_t *X, datapoint_array_t *C);
float dist(dp_t *x, datapoint_array_t *C);

/*
 * Run the k-means|| sample algorithm. C expects the initial datapoints to be
 * chosen.
 */
datapoint_array_t *kmeans_parallel_init(dps_t *X, int k);

/*
 * Functions to run k-means++. Not pretty, i know.
 */
void kmeanspp_init(datapoint_array_t *X, datapoint_array_t *C, int k);
void kmeanspp_impl(dps_t *X, datapoint_array_t *C);

float costpp(datapoint_array_t *X, datapoint_array_t *C);
