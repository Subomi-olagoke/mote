# mote — a tiny language model, run from scratch in C.
#
#   make          optimized single-thread build   -> ./mote
#   make omp      multi-threaded (needs OpenMP)    -> ./mote
#   make quantize the fp32 -> int8 converter       -> ./quantize
#   make debug    warnings + sanitizers            -> ./mote
#   make clean

CC      ?= cc
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra
LDLIBS  := -lm
SRC     := $(wildcard src/*.c)
BIN     := mote

$(BIN): $(SRC)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

.PHONY: omp quantize debug clean
omp: $(SRC)
	$(CC) $(CFLAGS) -fopenmp -o $(BIN) $(SRC) $(LDLIBS)

# the converter reuses the engine's quantization core
quantize: tools/quantize.c src/quant.c
	$(CC) $(CFLAGS) -o $@ tools/quantize.c src/quant.c $(LDLIBS)

# emit a model as a const C array to link into firmware (weights in flash)
blob2c: tools/blob2c.c
	$(CC) $(CFLAGS) -o $@ tools/blob2c.c

debug: $(SRC)
	$(CC) -O1 -g -std=c11 -Wall -Wextra -fsanitize=address,undefined \
	      -o $(BIN) $(SRC) $(LDLIBS)

clean:
	rm -f $(BIN) quantize blob2c
