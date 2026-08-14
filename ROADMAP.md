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

## 2. Quantize — done (int8 and int4)

Get the model small enough to be interesting on constrained hardware, with the
quantized arithmetic written by hand rather than pulled from a library.

- [x] Q8_0-style symmetric int8 weights with per-group scales
- [x] An integer-aware `matmul` that quantizes activations and folds scales in
- [x] A converter (`tools/quantize.c`) that writes mote's own `.mq` format
- [x] One binary auto-detects fp32 vs `.mq` and runs either
- [x] Measured: 3.6x smaller on the 15M model, ~0.0003 RMS/weight, quality held
- [x] int4: two weights per byte, signed-scale Q4_0-style rounding, its own
      nibble-unpacking matmul. 6.4x smaller than fp32; perplexity 2.640 vs
      2.376 fp32 on held-out TinyStories (int8: 2.379). Scheme chosen by
      measurement — see the README's quantization section.
- [ ] Quantize the KV cache (activation memory, matters on small devices)

## 3. Onto a microcontroller — the point

Run generation on a chip. The model that fits will be small, a story-teller, not
a chatbot, and that is exactly the charm.

- [ ] Pick a target with enough RAM (an ESP32-S3 with PSRAM is the honest
      starting line; smaller parts for the smallest models)
- [x] Load the model from a memory blob, not a file
      (`build_transformer_from_blob`); the portable core has no files or
      platform calls. Desktop mmaps, an MCU points at flash, iOS at the bundle.
- [x] Emit the model as a linkable flash blob (`blob2c`) and run the engine from
      it with zero files opened. Verified on the host with stories260K: weights
      in a `const` array (flash), coherent generation, no filesystem.
- [ ] Fixed buffers behind a static arena, no heap on the hot path
- [ ] Stream generated tokens out over USB serial
- [ ] Report the real numbers: tokens/sec and memory, no hand-waving

## Non-goals

- Beating llama.cpp on speed. This is a legibility-first engine.
- Training. `mote` only does inference; the weights come pretrained.
- Chat-scale models. The whole exercise is about how small you can go.
