plot_birch <- function() {
	precenters <- read.csv("preoutput.csv", header=FALSE);
	reduced <- read.csv("reducedoutput.csv", header=FALSE);
	centers <- read.csv("output.csv", header=FALSE);
	points  <- read.csv("datasets/birch3.txt", header=FALSE);
	#c2 <- read.csv("center2.csv", header=FALSE)
	p <- ggplot(data = d) + geom_point(aes(X, Y), size = 0.2)
	#p + geom_point(data = c2, aes(V1, V2), colour = "blue", size= 2) + geom_point(data = c, aes(V1, V2), colour = "red", size = 2)
	p <- p + geom_point(data = precenters, aes(V1, V2), colour = "blue", size = 1.0);
	p <- p + geom_point(data = reduced, aes(V1, V2), colour = "green", size = 1.0);
	p <- p + geom_point(data = centers, aes(V1, V2), colour = "red", size = 2);
	ggsave(filename = "plot.png", plot = p, width = 30, height=20, 
		   units = "cm");
}

cachesize_plot <- function() {
	library("ggplot2");

	sharedmem  <- 48*1024;
	blocksizes <- c(64, 128, 192, 256);

	#p <- ggplot() + ylim(0, sharedmem) + xlim(0, 200);
	p <- ggplot() + xlim(0, 200);
	p <- p + geom_abline(aes(intercept = 0, slope=blocksizes * 4, colour = factor(blocksizes)));

	p <- p + labs(title = "Test", x = "dimensions", y = "cachesize");

	p <- p + scale_y_continuous(limits = c(0, sharedmem), breaks = seq(0, 48*1024, 2048));

	p;
}

redsum <- function() {
	val <- (d$X - d$X[1])^2 + (d$Y - d$Y[1])^2
	for(i in seq(0,97, 1)) {
		mysum <- sum(val[((i*1024)+1):((((i+1)*1024)))])
		cat("sum(val[", (i*1024)+1, ":", ((i+1)*1024) %% 1, "]) = ", mysum, "\n");
	}
}

dimcache <- function() {
	cache <- 48*1024;

	maxdim = cache/(2*4)

	d <- numeric();
	cachecount <- numeric();
	for(dimension in seq(2, 2000, 2)) {
		d <- c(d, dimension);
		cachecount <- c(cachecount, floor(cache/(4*dimension)));
	}

	cachestore <- data.frame(Dim=d, Count=cachecount);

	#ggplot(data=cachestore) + geom_step(aes(x=Dim, y=Count)) + xlim(0,500);

	return(cachestore);
}
