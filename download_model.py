#!/usr/bin/env python3
"""
GOT-OCR Model Download Script
Downloads model weights from HuggingFace Hub
"""

import os
import sys
from pathlib import Path


def download_model():
    """Download GOT-OCR model from HuggingFace"""

    model_dir = Path("weights/GOT-OCR2_0")
    model_file = model_dir / "model.safetensors"

    # Check if already exists
    if model_file.exists():
        size = model_file.stat().st_size
        if size > 1_000_000_000:  # > 1GB
            print(f"Model already exists ({size / 1e9:.2f} GB). Skipping download.")
            return True
        else:
            print("Existing file is incomplete. Re-downloading...")
            model_file.unlink()

    print("=" * 50)
    print("GOT-OCR Model Downloader")
    print("=" * 50)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("\nInstalling huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import hf_hub_download

    print("\nDownloading GOT-OCR model from HuggingFace...")
    print("Repository: stepfun-ai/GOT-OCR2_0")
    print("File: model.safetensors (1.4GB)")
    print()

    try:
        # Download using huggingface_hub
        downloaded_path = hf_hub_download(
            repo_id="stepfun-ai/GOT-OCR2_0",
            filename="model.safetensors",
            local_dir=str(model_dir),
            local_dir_use_symlinks=False
        )

        print(f"\nDownload complete!")
        print(f"Model saved to: {downloaded_path}")

        # Verify
        size = Path(downloaded_path).stat().st_size
        print(f"File size: {size / 1e9:.2f} GB")

        if size < 1_000_000_000:
            print("Warning: File size seems too small. Download may have failed.")
            return False

        print("\nSetup complete! You can now run the OCR service.")
        return True

    except Exception as e:
        print(f"\nError downloading model: {e}")
        print("\nManual download instructions:")
        print("1. Visit: https://huggingface.co/stepfun-ai/GOT-OCR2_0")
        print("2. Download 'model.safetensors'")
        print("3. Place it in: weights/GOT-OCR2_0/")
        return False


if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
