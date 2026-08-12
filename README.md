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

Or run it quantized, about 4x smaller and noticeably faster:

```bash
make quantize                                        # builds the converter
./quantize models/stories15M.bin models/stories15M.mq
./mote models/stories15M.mq -i "Once upon a time"    # int8, same output quality
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
| `src/quant.c` | int8 quantization: the pack, dequant, and the int8 matmul |
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

`mote` also runs int8. The converter packs each big weight matrix into 8-bit
integers with one fp32 scale per group of values (symmetric "Q8_0" quantization),
and the runtime does the matmul in int8, quantizing each activation on the fly
and folding the scales back in per group. The norm gains stay fp32.

One binary runs both: a checkpoint's first bytes say whether it is fp32 or `.mq`,
and the engine picks the path. On the 15M model, int8 is about **3.6x smaller**
on disk (61 MB to 17 MB) with a per-weight RMS error near 0.0003, and generates
the same quality of text. Because the weights are a quarter of the size, it is
also markedly faster to run, memory bandwidth is the bottleneck, not arithmetic.

## Roadmap

`mote` is being built in three milestones, from the general to the absurd. See
[ROADMAP.md](ROADMAP.md).

1. **Inference in pure C** — done. This is what you are running now.
2. **Quantize** the weights to 8-bit, with the math by hand — done, about 4x
   smaller. int4 for the weight-heavy layers is still open.
3. **Onto a microcontroller** — run it on a chip, not a computer.

## Acknowledgements

The pretrained weights are the **TinyStories** models trained by Andrej Karpathy,
and `mote` reads the compact checkpoint format from his `llama2.c` project. That
format and those models are what make a from-scratch engine testable on day one.
The transformer architecture is Meta AI's Llama. The code here is my own
implementation.

## License

MIT. See [LICENSE](LICENSE).
