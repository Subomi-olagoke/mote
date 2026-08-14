#!/usr/bin/env python3
"""
tools/convert_hf.py — convert a HuggingFace Qwen2 checkpoint into mote's .mq format.

mote's engine is Llama-shaped. Qwen2 is close but differs in three ways this script
reconciles:

  1. Attention bias. Qwen has a bias on q/k/v; Llama does not. These are written to
     the .mq (v2) and added in the forward pass before RoPE.
  2. RoPE base and norm epsilon. Qwen uses theta=1e6 and eps=1e-6; they go in the
     header so the runtime uses the right constants.
  3. Rotary layout. Every HF checkpoint stores q_proj/k_proj in the "rotate-half"
     arrangement, while mote (following llama2.c) rotates interleaved pairs. So the
     q/k weights AND their biases are permuted back to interleaved here. Miss this
     and the model emits fluent-looking nonsense.

The big matrices are quantized to symmetric int8 (default) or packed int4
(--bits 4) with one fp32 scale per group, exactly what mote's runtime
dequantizes. int4 uses a per-group clipped-scale search: shrinking the scale
clips the group's outlier but gives every other weight finer steps, and the
scale with the least round-trip error wins. Norm gains and biases stay fp32.

  python3 tools/convert_hf.py <hf_model_dir> <out.mq> [--gs 64] [--bits 8] [--seqlen 2048]
"""
import argparse
import glob
import json
import os
import struct

import numpy as np

MQ_MAGIC = 0x65746F6D
MQ_VERSION = 2
MQ_HEADER = 256


def load_hf(path):
    """Load every safetensors shard as fp32 numpy, plus the config."""
    from safetensors import safe_open  # ships with transformers
    cfg = json.load(open(os.path.join(path, "config.json")))
    sd = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, framework="pt") as st:   # "pt" handles bf16 -> float()
            for k in st.keys():
                sd[k] = st.get_tensor(k).float().numpy().astype(np.float32)
    if not sd:
        raise SystemExit(f"no .safetensors found in {path}")
    return sd, cfg


def permute_reverse(w, n_heads, head_size):
    """HF rotate-half layout -> mote interleaved layout, on the output dimension.
    w is (n_heads*head_size, in) or (n_heads*head_size,). This is the exact inverse
    of the permutation HF applies when importing Llama-family weights."""
    d1 = w.shape[0]
    rest = int(np.prod(w.shape[1:])) if w.ndim > 1 else 1
    v = w.reshape(n_heads, 2, head_size // 2, rest)
    v = v.transpose(0, 2, 1, 3)
    return v.reshape(d1) if w.ndim == 1 else v.reshape(d1, rest)


def quantize_flat(w, gs):
    """Symmetric int8, one fp32 scale per group of gs, row-major, matching the
    layout mote's matmul_q8 expects. Returns (int8 bytes, fp32 scales)."""
    flat = np.ascontiguousarray(w, dtype=np.float32).ravel()
    assert flat.size % gs == 0, f"group size {gs} does not divide {flat.size}"
    g = flat.reshape(-1, gs)
    scale = np.abs(g).max(axis=1) / 127.0
    scale[scale == 0.0] = 1.0
    q = np.clip(np.rint(g / scale[:, None]), -127, 127).astype(np.int8)
    return q.tobytes(), scale.astype(np.float32).tobytes()


def quantize_flat_q4(w, gs):
    """int4 in [-8, 7], two per byte, exactly mote's quantize_q4: the scale is
    the group's largest-magnitude weight over -8 (keeping its sign, so that
    weight lands exactly on -8 and the rest get finer steps), plus a small
    shrunken-scale search kept only when it lowers round-trip error. Packing
    matches the runtime: within a group, byte k holds w[k] (low nibble) and
    w[k + gs/2] (high nibble). Returns (packed bytes, fp32 scales). Chunked so
    the embedding matrix does not need candidate-count copies of itself."""
    assert gs % 2 == 0
    flat = np.ascontiguousarray(w, dtype=np.float32).ravel()
    assert flat.size % gs == 0, f"group size {gs} does not divide {flat.size}"
    g = flat.reshape(-1, gs)
    half = gs // 2

    packed = np.empty((len(g), half), dtype=np.uint8)
    scales = np.empty(len(g), dtype=np.float32)
    chunk = 1 << 20                          # groups per pass
    for lo in range(0, len(g), chunk):
        gc = g[lo:lo + chunk]
        m = np.take_along_axis(gc, np.abs(gc).argmax(axis=1)[:, None], axis=1)[:, 0]
        best_err = np.full(len(gc), np.inf, dtype=np.float32)
        best_scale = np.zeros(len(gc), dtype=np.float32)
        best_q = np.zeros(gc.shape, dtype=np.int8)
        for c in range(5):
            scale = (m / -8.0) * (1.0 - 0.05 * c)
            safe = np.where(scale != 0.0, scale, 1.0)
            q = np.clip(np.rint(gc / safe[:, None]), -8, 7).astype(np.int8)
            err = ((gc - q * scale[:, None]) ** 2).sum(axis=1)
            better = err < best_err
            best_err[better] = err[better]
            best_scale[better] = scale[better]
            best_q[better] = q[better]
        scales[lo:lo + chunk] = best_scale
        qlo = best_q[:, :half].astype(np.int16)
        qhi = best_q[:, half:].astype(np.int16)
        packed[lo:lo + chunk] = ((qlo & 0xF) | ((qhi & 0xF) << 4)).astype(np.uint8)
    return packed.tobytes(), scales.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--gs", type=int, default=64)
    ap.add_argument("--bits", type=int, default=8, choices=(8, 4),
                    help="weight width; 4 halves the file again at some quality cost")
    ap.add_argument("--seqlen", type=int, default=2048,
                    help="cap the context (bounds the KV cache); RoPE is unaffected")
    args = ap.parse_args()

    sd, cfg = load_hf(args.model_dir)
    D = cfg["hidden_size"]
    HD = cfg["intermediate_size"]
    L = cfg["num_hidden_layers"]
    NH = cfg["num_attention_heads"]
    NKV = cfg["num_key_value_heads"]
    V = cfg["vocab_size"]
    head_size = D // NH
    kv = NKV * head_size
    seq_len = min(args.seqlen, cfg.get("max_position_embeddings", args.seqlen))
    rope_theta = float(cfg.get("rope_theta", 10000.0))
    rms_eps = float(cfg.get("rms_norm_eps", 1e-5))
    shared = bool(cfg.get("tie_word_embeddings", False))
    gs = args.gs
    if D % gs or HD % gs:
        raise SystemExit(f"gs {gs} must divide dim {D} and hidden {HD}")
    if NH * head_size != D:
        raise SystemExit("mote requires dim == n_heads * head_size")

    def g(name):
        return sd[name]

    # ---- assemble fp32 side (norms + biases), per-layer stacked ----
    rms_att = np.stack([g(f"model.layers.{l}.input_layernorm.weight") for l in range(L)])
    rms_ffn = np.stack([g(f"model.layers.{l}.post_attention_layernorm.weight") for l in range(L)])
    rms_final = g("model.norm.weight")
    bq = np.stack([permute_reverse(g(f"model.layers.{l}.self_attn.q_proj.bias"), NH, head_size) for l in range(L)])
    bk = np.stack([permute_reverse(g(f"model.layers.{l}.self_attn.k_proj.bias"), NKV, head_size) for l in range(L)])
    bv = np.stack([g(f"model.layers.{l}.self_attn.v_proj.bias") for l in range(L)])

    print(f"config: dim={D} hidden={HD} layers={L} heads={NH}/{NKV} vocab={V} "
          f"seq={seq_len} theta={rope_theta:g} eps={rms_eps:g} shared={shared} "
          f"gs={gs} bits={args.bits}", flush=True)

    with open(args.out, "wb") as f:
        # header: 64 int32 slots (256 bytes). Stamp the lowest version that can
        # express the file, so int8 output still loads on a v2 runtime.
        h = [0] * (MQ_HEADER // 4)
        h[0] = MQ_MAGIC; h[1] = 3 if args.bits == 4 else 2
        h[2] = D; h[3] = HD; h[4] = L; h[5] = NH; h[6] = NKV; h[7] = V; h[8] = seq_len
        h[9] = gs; h[10] = 1 if shared else 0
        h[11] = 1  # has_qkv_bias
        f.write(struct.pack("<12i", *h[:12]))
        f.write(struct.pack("<f", rope_theta))   # slot 12 as float
        f.write(struct.pack("<f", rms_eps))      # slot 13 as float
        f.write(struct.pack("<i", args.bits))    # slot 14
        f.write(b"\x00" * (MQ_HEADER - 15 * 4))

        # fp32 norms, then fp32 biases
        for a in (rms_att, rms_ffn, rms_final, bq, bk, bv):
            f.write(np.ascontiguousarray(a, dtype=np.float32).tobytes())

        # quantized tensors, in the order the loader maps them
        def wq(mat):
            if args.bits == 4:
                qb, sb = quantize_flat_q4(mat, gs)
            else:
                qb, sb = quantize_flat(mat, gs)
            f.write(qb); f.write(sb)

        wq(g("model.embed_tokens.weight"))                                    # tokens
        wq(np.stack([permute_reverse(g(f"model.layers.{l}.self_attn.q_proj.weight"), NH, head_size) for l in range(L)]))
        wq(np.stack([permute_reverse(g(f"model.layers.{l}.self_attn.k_proj.weight"), NKV, head_size) for l in range(L)]))
        wq(np.stack([g(f"model.layers.{l}.self_attn.v_proj.weight") for l in range(L)]))
        wq(np.stack([g(f"model.layers.{l}.self_attn.o_proj.weight") for l in range(L)]))
        wq(np.stack([g(f"model.layers.{l}.mlp.gate_proj.weight") for l in range(L)]))   # w1
        wq(np.stack([g(f"model.layers.{l}.mlp.down_proj.weight") for l in range(L)]))   # w2
        wq(np.stack([g(f"model.layers.{l}.mlp.up_proj.weight") for l in range(L)]))     # w3
        if not shared:
            wq(g("lm_head.weight"))

    mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}  ({mb:.1f} MB, int8 gs={gs})", flush=True)


if __name__ == "__main__":
    main()
