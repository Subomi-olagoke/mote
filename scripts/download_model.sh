#!/usr/bin/env bash
# Fetch a pretrained TinyStories model and the tokenizer into models/.
#
#   ./scripts/download_model.sh          # 15M (default), good on any laptop
#   ./scripts/download_model.sh 42M
#   ./scripts/download_model.sh 110M
#
# The weights are Andrej Karpathy's TinyStories models; see the README.
set -euo pipefail

size="${1:-15M}"
here="$(cd "$(dirname "$0")/.." && pwd)"
models="$here/models"
mkdir -p "$models"

base="https://huggingface.co/karpathy/tinyllamas/resolve/main"
case "$size" in
  15M|42M|110M) url="$base/stories${size}.bin" ;;
  *) echo "unknown size '$size' (use 15M, 42M, or 110M)" >&2; exit 1 ;;
esac

echo "→ tokenizer"
curl -fL# -o "$models/tokenizer.bin" \
  https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin

echo "→ stories${size}.bin"
curl -fL# -o "$models/stories${size}.bin" "$url"

echo
echo "done. try:"
echo "  ./mote models/stories${size}.bin -i \"Once upon a time\""
