CC = gcc
CFLAGS = -O3 -fomit-frame-pointer -std=c11 -Wall -Wextra -pedantic
INC = -I src
LIB = -L lib

CSVCONVERT = build/csvconvert.o

csvconvert: $(CSVCONVERT)
	$(CC) $(CFLAGS) $(LIB) -o $@ $^

build/%.o: src/%.c
	mkdir -p build
	$(CC) -c $(CFLAGS) $(INC) $< -o $@

clean:
	rm -rf build
	rm -f csvconvert
