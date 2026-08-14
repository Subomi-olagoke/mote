# mote

A tiny language model, run from scratch in C.

`mote` is a complete transformer inference engine in a few hundred lines of
dependency-free C. No PyTorch, no NumPy, no BLAS, no framework. It loads a
pretrained model and generates text, and the whole forward pass, attention,
rotary embeddings, the KV cache, SwiGLU, sampling, is written out by hand so you
can read every step.

The name is the point. A mote is the smallest thing you can still see. The goal
of this project is to shrink a working language model down until it fits somewhere
absurd: a laptop with nothing installed, and eventually a microcontroller.

```
$ ./mote models/stories15M.bin -i "Once upon a time, there was a little robot"

Once upon a time, there was a little robot. The robot liked to crawl. One day,
the robot went to the park. He saw a big tree... At the top, he saw a big party.
The party was for the robot. The robot and the bird were friends. They played
and had fun.

[158 tokens, 109.4 tok/s]
```

## Quickstart

```bash
make                              # builds ./mote, needs only a C compiler
./scripts/download_model.sh       # fetches a 15M-param model + tokenizer (~58MB)
./mote models/stories15M.bin -i "Once upon a time"
```

Or run it quantized, 4x smaller (int8) or 6x smaller (int4) and noticeably faster:

```bash
make quantize                                        # builds the converter
./quantize models/stories15M.bin models/stories15M.mq
./mote models/stories15M.mq -i "Once upon a time"    # int8, same output quality
./quantize models/stories15M.bin models/stories15M_q4.mq 32 4
./mote models/stories15M_q4.mq -i "Once upon a time" # int4, still coherent
```

Options:

```
-z <path>   tokenizer file (default models/tokenizer.bin)
-t <float>  temperature, 0 = greedy/deterministic (default 1.0)
-p <float>  top-p / nucleus sampling (default 0.9, 0 disables)
-n <int>    number of steps (default 256)
-i <string> prompt
-s <int>    rng seed, for reproducible runs
```

Larger models (`42M`, `110M`) are one argument away:
`./scripts/download_model.sh 110M`.

## How it works

The engine is small enough to hold in your head:

| file | what it does |
|---|---|
| `src/model.c` | memory-maps a checkpoint and points the weights at it |
| `src/forward.c` | one token through the network, the core of the whole thing |
| `src/tokenizer.c` | byte-pair encode/decode between text and token ids |
| `src/sampler.c` | greedy, temperature, and top-p sampling |
| `src/quant.c` | int8 and int4 quantization: the pack, dequant, and integer matmuls |
| `src/mote.c` | the CLI and the generation loop |
| `tools/quantize.c` | converts an fp32 checkpoint to the quantized `.mq` format |

The model is a standard Llama-style decoder: pre-norm blocks, each with
grouped-query attention and a SwiGLU feed-forward, RMSNorm throughout, and
rotary position embeddings applied directly to the query and key vectors. A KV
cache keeps each step to attending over history rather than recomputing it.
Weights are memory-mapped, so a model is paged in on demand and startup is
instant.

There is exactly one performance-critical loop, the matrix-vector `matmul`, and
it is the only thing touched by optional multithreading:

```bash
make omp      # OpenMP build, if your compiler has it
```

## Quantization

`mote` also runs quantized weights, at two widths. int8 packs each big weight
matrix into 8-bit integers with one fp32 scale per group of values (symmetric
"Q8_0" quantization). int4 goes further: two weights per byte in [-8, 7], where
the scale carries the sign of the group's largest weight so that weight lands
exactly on -8 and every other weight gets a finer step (the "Q4_0" idea). In
both, the runtime quantizes each activation to int8 on the fly and the matmul
is integer work with the scales folded back in per group. The norm gains stay
fp32.

One binary runs all of it: a checkpoint's first bytes say whether it is fp32 or
`.mq`, and the header says how wide the weights are. And the quality is
measured, not asserted: `make perplexity` scores a model over held-out text
(TinyStories validation data, ~17k tokens). On the 15M model:

| weights | size | perplexity | cost |
|---|---|---|---|
| fp32 | 61 MB | 2.376 | — |
| int8 (gs 32) | 17 MB | 2.379 | +0.1% |
| int4 (gs 32) | 9.5 MB | 2.640 | +11% |

Getting the int4 number down was an exercise in measuring rather than trusting
intuition: a per-group clipped-scale search *lowers weight-space RMS error* and
*worsens* perplexity, because the weights it clips are precisely the ones that
matter. The signed-scale trick above beat every clipping scheme tried; the
small scale search kept in the converter is worth ~0.1% on top. Weight RMS
error and model quality are different objectives, only one of them is the point.

Because quantized weights are a quarter (or an eighth) of the size, quantized
models are also markedly faster to run, memory bandwidth is the bottleneck, not
arithmetic.

## In a browser

The engine also compiles to WebAssembly — same C files, one more host:

```bash
./web/build.sh                    # needs emscripten; emits web/mote.js + .wasm
python3 -m http.server 8000       # serve from the repo root
# open http://localhost:8000/web/  (append ?auto to run a benchmark prompt)
```

The whole engine is ~72KB of wasm; the model download is the only heavy part.
Measured with the int4 Qwen2.5-0.5B (309 MB): 13.6 tok/s in desktop Chromium
and 6.5 tok/s in iPhone Safari, single-threaded scalar — no NEON in wasm, no
threads yet. Slower than the native app, but it is a real chat running from a
URL with nothing installed.

## Roadmap

`mote` is being built in three milestones, from the general to the absurd. See
[ROADMAP.md](ROADMAP.md).

1. **Inference in pure C** — done. This is what you are running now.
2. **Quantize** the weights, with the math by hand — done at both widths: int8
   (4x smaller, quality intact) and int4 (6x smaller, quality measured above).
3. **Onto a microcontroller** — run it on a chip, not a computer.

## Acknowledgements

The pretrained weights are the **TinyStories** models trained by Andrej Karpathy,
and `mote` reads the compact checkpoint format from his `llama2.c` project. That
format and those models are what make a from-scratch engine testable on day one.
The transformer architecture is Meta AI's Llama. The code here is my own
implementation.

## License

MIT. See [LICENSE](LICENSE).
