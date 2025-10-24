"""
Download DeepSeek-OCR model from Hugging Face and convert to MLX format.

This script:
1. Downloads the pre-trained model from deepseek-ai/DeepSeek-OCR
2. Converts PyTorch weights to MLX format
3. Saves in a format ready for inference

Usage:
    python download_and_convert.py

Requirements:
    pip install huggingface_hub safetensors numpy mlx
"""

import os
import sys
from pathlib import Path
import json
import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    print("ERROR: MLX not installed. Install with: pip install mlx")
    sys.exit(1)

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    print("ERROR: huggingface_hub not installed. Install with: pip install huggingface_hub")
    sys.exit(1)

try:
    from safetensors import safe_open
except ImportError:
    print("ERROR: safetensors not installed. Install with: pip install safetensors")
    sys.exit(1)


MODEL_REPO = "deepseek-ai/DeepSeek-OCR"
OUTPUT_DIR = Path(__file__).parent / "weights"


def download_model():
    """Download model from Hugging Face."""
    print("=" * 80)
    print("Downloading DeepSeek-OCR from Hugging Face")
    print("=" * 80)
    print()
    print(f"Model: {MODEL_REPO}")
    print(f"Destination: {OUTPUT_DIR}")
    print()
    print("This may take a while (model is ~10GB)...")
    print()

    try:
        # Download entire model repo
        local_dir = snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=OUTPUT_DIR,
            local_dir_use_symlinks=False,
        )

        print()
        print(f"✅ Model downloaded to: {local_dir}")
        return Path(local_dir)

    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        print()
        print("Troubleshooting:")
        print("1. Check internet connection")
        print("2. Verify Hugging Face access (may need: huggingface-cli login)")
        print("3. Check disk space (~15GB required)")
        sys.exit(1)


def load_pytorch_weights(model_dir):
    """Load PyTorch weights from safetensors."""
    print()
    print("=" * 80)
    print("Loading PyTorch Weights")
    print("=" * 80)
    print()

    weights = {}

    # Find all safetensors files
    safetensor_files = list(model_dir.glob("*.safetensors"))

    if not safetensor_files:
        print("⚠️  No .safetensors files found")
        print("Looking for pytorch_model.bin...")

        # Try .bin files as fallback
        bin_files = list(model_dir.glob("*.bin"))
        if bin_files:
            print("❌ Found .bin files but MLX requires .safetensors format")
            print("Please use a model with safetensors weights")
            sys.exit(1)
        else:
            print("❌ No weight files found")
            sys.exit(1)

    print(f"Found {len(safetensor_files)} safetensor file(s)")

    # Load all safetensor files
    for st_file in safetensor_files:
        print(f"Loading: {st_file.name}")
        with safe_open(st_file, framework="numpy") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)

    print(f"✅ Loaded {len(weights)} weight tensors")
    return weights


def convert_to_mlx(weights):
    """Convert PyTorch weights to MLX format."""
    print()
    print("=" * 80)
    print("Converting to MLX Format")
    print("=" * 80)
    print()

    mlx_weights = {}

    for key, value in weights.items():
        # Convert numpy to MLX array
        mlx_weights[key] = mx.array(value)

        if len(mlx_weights) % 100 == 0:
            print(f"Converted {len(mlx_weights)}/{len(weights)} tensors...")

    print(f"✅ Converted {len(mlx_weights)} tensors to MLX format")
    return mlx_weights


def save_mlx_weights(weights, output_dir):
    """Save MLX weights."""
    print()
    print("=" * 80)
    print("Saving MLX Weights")
    print("=" * 80)
    print()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save weights as npz (MLX can load this)
    weights_file = output_dir / "weights.npz"

    print(f"Saving to: {weights_file}")

    # Convert MLX arrays to numpy for saving
    numpy_weights = {k: np.array(v) for k, v in weights.items()}
    np.savez(str(weights_file), **numpy_weights)

    print(f"✅ Weights saved: {weights_file}")
    print(f"   Size: {weights_file.stat().st_size / (1024**3):.2f} GB")


def copy_config_and_tokenizer(model_dir, output_dir):
    """Copy config and tokenizer files."""
    print()
    print("=" * 80)
    print("Copying Configuration Files")
    print("=" * 80)
    print()

    output_dir = Path(output_dir)

    # Files to copy
    files_to_copy = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]

    for filename in files_to_copy:
        src = model_dir / filename
        if src.exists():
            dst = output_dir / filename
            import shutil
            shutil.copy(src, dst)
            print(f"✅ Copied: {filename}")
        else:
            print(f"⚠️  Not found: {filename}")


def main():
    """Main conversion pipeline."""
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "DeepSeek-OCR MLX Converter" + " " * 32 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Step 1: Download model
    model_dir = OUTPUT_DIR

    if model_dir.exists() and any(model_dir.glob("*.safetensors")):
        print(f"Model already downloaded at: {model_dir}")
        response = input("Re-download? (y/N): ").strip().lower()
        if response == 'y':
            model_dir = download_model()
    else:
        model_dir = download_model()

    # Step 2: Load PyTorch weights
    pytorch_weights = load_pytorch_weights(model_dir)

    # Step 3: Convert to MLX
    mlx_weights = convert_to_mlx(pytorch_weights)

    # Step 4: Save MLX weights
    mlx_output_dir = Path(__file__).parent / "weights_mlx"
    save_mlx_weights(mlx_weights, mlx_output_dir)

    # Step 5: Copy config files
    copy_config_and_tokenizer(model_dir, mlx_output_dir)

    # Done!
    print()
    print("=" * 80)
    print("✅ CONVERSION COMPLETE")
    print("=" * 80)
    print()
    print(f"MLX weights saved to: {mlx_output_dir}")
    print()
    print("Next steps:")
    print("1. Run inference: python run_inference.py --image your_image.jpg")
    print("2. Or use Python API:")
    print()
    print("   from deepseek_ocr_mlx import DeepSeekOCR")
    print("   model = DeepSeekOCR.from_pretrained('weights_mlx')")
    print("   result = model.infer('your_image.jpg', prompt='<image>\\nFree OCR.')")
    print()


if __name__ == "__main__":
    main()
