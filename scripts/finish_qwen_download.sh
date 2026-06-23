#!/bin/bash
# Once /tmp/qwen3-0.6b.safetensors is complete, symlink it into the HF cache
# so transformers / load_qwen3 finds it.

set -e
TGT=/Users/chris/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base
SNAP=$(ls -d "$TGT"/snapshots/* | head -1)
REF=$(cat "$TGT"/refs/main)
BLOBS="$TGT/blobs"

# wait for download to finish
while ! grep -q "^100 " /tmp/qwen3_dl.log 2>/dev/null; do
    sleep 5
done
sleep 2

# verify file size
SIZE=$(stat -f%z /tmp/qwen3-0.6b.safetensors 2>/dev/null || stat -c%s /tmp/qwen3-0.6b.safetensors)
echo "downloaded size: $SIZE bytes (expected ~1192135096)"

if [ "$SIZE" -lt 1000000000 ]; then
    echo "ERROR: download incomplete"
    exit 1
fi

# compute sha and link
SHA=$(shasum -a 256 /tmp/qwen3-0.6b.safetensors | cut -d' ' -f1)
BLOB_PATH="$BLOBS/$SHA"

if [ ! -e "$BLOB_PATH" ]; then
    mkdir -p "$BLOBS"
    mv /tmp/qwen3-0.6b.safetensors "$BLOB_PATH"
    echo "moved to $BLOB_PATH"
fi

# symlink in snapshot
SNAP_PATH="$SNAP/model.safetensors"
if [ ! -L "$SNAP_PATH" ]; then
    ln -s "../../blobs/$SHA" "$SNAP_PATH"
    echo "linked $SNAP_PATH"
fi

echo "OK - Qwen3-0.6B-Base ready"
ls -la "$SNAP_PATH"
