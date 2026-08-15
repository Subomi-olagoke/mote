#!/bin/sh
# web/build.sh — compile the engine to WebAssembly.
#
# Same C sources as every other target; the browser is just one more host for
# mote_lib. -msimd128 turns on the hand-written wasm SIMD dot products in
# quant.c/forward.c, and -pthread turns on the worker pool in parallel.c
# (pthreads map onto web workers; the pool is preallocated at link time).
# Threads need cross-origin isolation to get SharedArrayBuffer — serve with
# web/serve.py, which sends the two headers. Without them the engine still
# runs, single-threaded. Produces web/mote.js + web/mote.wasm.
set -e
cd "$(dirname "$0")/.."

emcc -O3 -std=c11 -msimd128 -pthread \
  src/model.c src/forward.c src/quant.c src/parallel.c \
  src/tokenizer.c src/sampler.c src/bpe.c src/mote_lib.c \
  -o web/mote.js \
  -sMODULARIZE=1 -sEXPORT_NAME=createMote \
  -sENVIRONMENT=worker \
  -sPTHREAD_POOL_SIZE=8 \
  -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=2147483648 \
  -sEXPORTED_FUNCTIONS=_mote_create,_mote_generate,_mote_embed,_mote_free,_mote_get_info,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,addFunction,removeFunction,UTF8ToString,HEAPU8,HEAP32,HEAPF32 \
  -sALLOW_TABLE_GROWTH=1

ls -la web/mote.js web/mote.wasm
