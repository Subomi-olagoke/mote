# mote — a tiny language model, run from scratch in C.
#
#   make          optimized single-thread build   -> ./mote
#   make omp      multi-threaded (needs OpenMP)    -> ./mote
#   make debug    warnings + sanitizers            -> ./mote
#   make clean

CC      ?= cc
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra
LDLIBS  := -lm
SRC     := $(wildcard src/*.c)
BIN     := mote

$(BIN): $(SRC)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

.PHONY: omp debug clean
omp: $(SRC)
	$(CC) $(CFLAGS) -fopenmp -o $(BIN) $(SRC) $(LDLIBS)

debug: $(SRC)
	$(CC) -O1 -g -std=c11 -Wall -Wextra -fsanitize=address,undefined \
	      -o $(BIN) $(SRC) $(LDLIBS)

clean:
	rm -f $(BIN)
