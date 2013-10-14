CC = gcc
CFLAGS = -O3 -fomit-frame-pointer -std=c11 -Wall -Wextra -pedantic -g
INC = -I src
LIB = -L lib

CSVCONVERT = build/csvconvert.o
KMEANS = build/kmeans.o build/datafile.o

all: kmeans csvconvert

kmeans: $(KMEANS)
	$(CC) $(CFLAGS) $(LIB) -o $@ $^

csvconvert: $(CSVCONVERT)
	$(CC) $(CFLAGS) $(LIB) -o $@ $^

build/%.o: src/%.c
	mkdir -p build
	$(CC) -c $(CFLAGS) $(INC) $< -o $@

clean:
	rm -rf build
	rm -f csvconvert
