"""
REAL weight loading - no placeholders.

This loads the actual DeepSeek-OCR weights from HuggingFace and converts to MLX.
"""

import os
import sys
from pathlib import Path
import json
from typing import Dict, Any
import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    print("ERROR: MLX not installed")
    print("Install: pip install mlx")
    sys.exit(1)

try:
    from huggingface_hub import snapshot_download, hf_hub_download
except ImportError:
    print("ERROR: huggingface_hub not installed")
    print("Install: pip install huggingface_hub")
    sys.exit(1)

try:
    from safetensors import safe_open
except ImportError:
    print("ERROR: safetensors not installed")
    print("Install: pip install safetensors")
    sys.exit(1)


def download_model_from_hf(model_id: str = "deepseek-ai/DeepSeek-OCR", cache_dir: str = None):
    """
    Download the COMPLETE model from HuggingFace.

    This downloads EVERYTHING - vision encoders + language model + all weights.
    """
    print(f"Downloading {model_id} from HuggingFace...")
    print("This downloads the COMPLETE model including the language model.")
    print("First download will take a while (~10-15GB)...")

    local_path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        local_dir_use_symlinks=False,
    )

    print(f"✅ Model downloaded to: {local_path}")
    return Path(local_path)


def load_safetensor_weights(model_dir: Path) -> Dict[str, np.ndarray]:
    """Load all weights from safetensors files."""
    print("\nLoading weights from safetensors...")

    weights = {}
    safetensor_files = list(model_dir.glob("*.safetensors"))

    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")

    print(f"Found {len(safetensor_files)} safetensor files")

    for st_file in safetensor_files:
        print(f"Loading: {st_file.name}")
        with safe_open(st_file, framework="numpy") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)

    print(f"✅ Loaded {len(weights)} weight tensors")
    return weights


def convert_weights_to_mlx(weights: Dict[str, np.ndarray]) -> Dict[str, mx.array]:
    """Convert numpy weights to MLX arrays."""
    print("\nConverting weights to MLX format...")

    mlx_weights = {}
    for key, value in weights.items():
        mlx_weights[key] = mx.array(value)

    print(f"✅ Converted {len(mlx_weights)} tensors to MLX")
    return mlx_weights


def load_model_weights(model_id: str = "deepseek-ai/DeepSeek-OCR", cache_dir: str = None):
    """
    Complete weight loading pipeline.

    Returns:
        Dict with:
        - 'weights': MLX weight dict
        - 'config': Model config
        - 'model_dir': Path to downloaded model
    """
    # Download
    model_dir = download_model_from_hf(model_id, cache_dir)

    # Load config
    config_file = model_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")

    with open(config_file) as f:
        config = json.load(f)

    # Load weights
    weights_np = load_safetensor_weights(model_dir)
    weights_mlx = convert_weights_to_mlx(weights_np)

    return {
        'weights': weights_mlx,
        'config': config,
        'model_dir': model_dir,
    }


if __name__ == "__main__":
    # Test weight loading
    print("=" * 80)
    print("Testing weight loading from HuggingFace")
    print("=" * 80)

    result = load_model_weights()

    print("\n" + "=" * 80)
    print("SUCCESS")
    print("=" * 80)
    print(f"Loaded {len(result['weights'])} weight tensors")
    print(f"Model dir: {result['model_dir']}")
    print("\nSample weight keys:")
    for i, key in enumerate(list(result['weights'].keys())[:10]):
        weight = result['weights'][key]
        print(f"  {key}: {weight.shape} {weight.dtype}")
