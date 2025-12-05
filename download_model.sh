#!/bin/bash
# ============================================
# GOT-OCR Model Download Script
# Downloads model weights from HuggingFace
# ============================================

set -e

MODEL_DIR="weights/GOT-OCR2_0"
MODEL_URL="https://huggingface.co/stepfun-ai/GOT-OCR2_0/resolve/main/model.safetensors"

echo "============================================"
echo "GOT-OCR Model Downloader"
echo "============================================"

# Check if model already exists
if [ -f "$MODEL_DIR/model.safetensors" ]; then
    FILE_SIZE=$(stat -f%z "$MODEL_DIR/model.safetensors" 2>/dev/null || stat -c%s "$MODEL_DIR/model.safetensors" 2>/dev/null)
    if [ "$FILE_SIZE" -gt 1000000000 ]; then
        echo "Model already exists ($FILE_SIZE bytes). Skipping download."
        exit 0
    else
        echo "Existing file is incomplete. Re-downloading..."
        rm -f "$MODEL_DIR/model.safetensors"
    fi
fi

# Create directory if not exists
mkdir -p "$MODEL_DIR"

echo ""
echo "Downloading GOT-OCR model (1.4GB)..."
echo "Source: $MODEL_URL"
echo ""

# Download with progress
if command -v wget &> /dev/null; then
    wget --progress=bar:force -O "$MODEL_DIR/model.safetensors" "$MODEL_URL"
elif command -v curl &> /dev/null; then
    curl -L --progress-bar -o "$MODEL_DIR/model.safetensors" "$MODEL_URL"
else
    echo "Error: wget or curl is required"
    exit 1
fi

echo ""
echo "Download complete!"
echo "Model saved to: $MODEL_DIR/model.safetensors"

# Verify download
FILE_SIZE=$(stat -f%z "$MODEL_DIR/model.safetensors" 2>/dev/null || stat -c%s "$MODEL_DIR/model.safetensors" 2>/dev/null)
echo "File size: $(numfmt --to=iec-i --suffix=B $FILE_SIZE 2>/dev/null || echo "$FILE_SIZE bytes")"

if [ "$FILE_SIZE" -lt 1000000000 ]; then
    echo "Warning: File size seems too small. Download may have failed."
    exit 1
fi

echo ""
echo "Setup complete! You can now run the OCR service."
