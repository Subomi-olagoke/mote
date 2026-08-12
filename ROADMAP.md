# Roadmap

The one-line goal: shrink a working language model until it runs somewhere it
has no business running. Three milestones, each a standalone, demonstrable thing.

## 1. Inference in pure C — done

A complete Llama-style transformer forward pass in dependency-free C, generating
coherent text from a pretrained model.

- [x] Memory-mapped checkpoint loader (the TinyStories / `llama2.c` format)
- [x] Forward pass: RMSNorm, RoPE, grouped-query attention, KV cache, SwiGLU
- [x] Byte-pair tokenizer: encode and decode
- [x] Sampling: greedy, temperature, top-p
- [x] CLI with reproducible seeds and tokens/sec reporting

## 2. Quantize — next

Get the model small enough to be interesting on constrained hardware, with the
quantized arithmetic written by hand rather than pulled from a library.

- [ ] Q8_0-style symmetric int8 weights with per-group scales
- [ ] An integer-aware `matmul` that dequantizes on the fly
- [ ] A converter that turns an fp32 checkpoint into mote's own quantized format
- [ ] Measure the quality/size tradeoff honestly (perplexity vs bytes)
- [ ] Explore sub-8-bit (int4) for the weight-heavy layers

## 3. Onto a microcontroller — the point

Run generation on a chip. The model that fits will be small, a story-teller, not
a chatbot, and that is exactly the charm.

- [ ] Pick a target with enough RAM (an ESP32-S3 with PSRAM is the honest
      starting line; smaller parts for the smallest models)
- [ ] Replace file mmap with weights in flash / external PSRAM
- [ ] Strip the desktop-only bits, fixed buffers, no dynamic allocation on the
      hot path
- [ ] Stream generated tokens out over USB serial
- [ ] Report the real numbers: tokens/sec and memory, no hand-waving

## Non-goals

- Beating llama.cpp on speed. This is a legibility-first engine.
- Training. `mote` only does inference; the weights come pretrained.
- Chat-scale models. The whole exercise is about how small you can go.
