#!/usr/bin/env python3
"""
tools/convert_tokenizer.py — pack a HuggingFace byte-level BPE tokenizer (Qwen2,
GPT-2 family) into a compact binary the mote C runtime loads (.mtok).

Byte-level BPE is a different algorithm from mote's original SentencePiece-style
tokenizer, so this is a new format. The trick that keeps the C side simple: every
merge is stored in ID space as (left_id, right_id) -> merged_id, and every raw byte
maps to the id of its 1-char byte-encoded token. So the encoder never hashes
strings except to spot special tokens.

  python3 tools/convert_tokenizer.py <hf_model_dir> <out.mtok>
"""
import json
import os
import struct
import sys


def bytes_to_unicode():
    """GPT-2's reversible byte<->unicode map. Printable bytes map to themselves;
    the rest map to U+0100.. so no token ever contains a control byte or space."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def main():
    model_dir, out = sys.argv[1], sys.argv[2]
    tj = json.load(open(os.path.join(model_dir, "tokenizer.json")))
    cfg = json.load(open(os.path.join(model_dir, "tokenizer_config.json")))

    vocab = tj["model"]["vocab"]              # token_str -> id (base BPE vocab)
    merges = tj["model"]["merges"]            # ["left right", ...] rank = index
    added = tj.get("added_tokens", [])        # special tokens
    # total vocab size (includes reserved/padding ids) from config if present
    total = None
    for f in ("config.json",):
        p = os.path.join(model_dir, f)
        if os.path.exists(p):
            total = json.load(open(p)).get("vocab_size")
    if total is None:
        total = max(list(vocab.values()) + [t["id"] for t in added]) + 1

    # id -> string table (byte-encoded space), filled from base vocab + specials
    id2str = [b""] * total
    for tok, i in vocab.items():
        if i < total:
            id2str[i] = tok.encode("utf-8")
    specials = []
    eos_str = cfg.get("eos_token")
    eos_id = -1
    for t in added:
        i, content = t["id"], t["content"]
        if i < total:
            id2str[i] = content.encode("utf-8")
        specials.append((content.encode("utf-8"), i))
        if content == eos_str:
            eos_id = i

    # byte -> id of its 1-char encoded token
    b2u = bytes_to_unicode()
    byte_to_id = [0] * 256
    for b in range(256):
        s = b2u[b]
        if s not in vocab:
            raise SystemExit(f"byte {b} ({s!r}) not in vocab; not a byte-level BPE tokenizer?")
        byte_to_id[b] = vocab[s]

    # merges as (left_id, right_id, merged_id)
    trip = []
    for m in merges:
        parts = m.split(" ") if isinstance(m, str) else list(m)
        if len(parts) != 2:
            raise SystemExit(f"bad merge line: {m!r}")
        l, r = parts
        merged = l + r
        if l not in vocab or r not in vocab or merged not in vocab:
            # a handful of merges can reference tokens missing from vocab; skip them
            continue
        trip.append((vocab[l], vocab[r], vocab[merged]))

    with open(out, "wb") as f:
        f.write(b"MTOK")
        f.write(struct.pack("<6i", 1, total, len(trip), len(specials), eos_id, -1))
        f.write(struct.pack("<256i", *byte_to_id))
        for s in id2str:
            f.write(struct.pack("<H", len(s))); f.write(s)
        for l, r, m in trip:
            f.write(struct.pack("<3i", l, r, m))
        for s, i in specials:
            f.write(struct.pack("<H", len(s))); f.write(s); f.write(struct.pack("<i", i))

    print(f"wrote {out}: vocab={total} merges={len(trip)} specials={len(specials)} "
          f"eos={eos_id} ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
