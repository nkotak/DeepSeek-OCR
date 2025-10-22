# DeepSeek-OCR MLX Migration - Comprehensive Technical Implementation Plan

## Executive Summary

**Objective:** Migrate DeepSeek-OCR from PyTorch/CUDA to MLX for Apple Silicon
**Timeline:** 2-3 weeks (55-70 hours total)
**Success Criteria:** <1% output difference vs PyTorch, working inference on Apple Silicon
**Risk Level:** Low-Medium (95%+ success probability)
**Team Size:** 1-2 developers

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Phase 1: Environment Setup](#phase-1-environment-setup--validation)
3. [Phase 2: Core Utilities](#phase-2-core-mlx-utilities-implementation)
4. [Phase 3: SAM Encoder](#phase-3-sam-vision-encoder-migration)
5. [Phase 4: CLIP Encoder](#phase-4-clip-vision-encoder-migration)
6. [Phase 5: MLP Projector](#phase-5-mlp-projector-migration)
7. [Phase 6: Main Model](#phase-6-main-model-integration)
8. [Phase 7: Inference Engine](#phase-7-inference-engine-replacement)
9. [Phase 8: Testing & Validation](#phase-8-testing--validation)
10. [Risk Management](#risk-management)
11. [Success Metrics](#success-metrics)

---

## Architecture Overview

### Current Architecture (PyTorch/CUDA)
```
Input Image (PIL)
    ↓
DeepseekOCRProcessor (image_process.py)
    ├─ Dynamic preprocessing (crop/tile)
    ├─ Normalization (mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    └─ Output: [pixel_values, images_crop, images_spatial_crop]
    ↓
Vision Encoders (Parallel Processing)
    ├── SAM ViT-B (sam_vary_sdpa.py)
    │   ├─ PatchEmbed: 16x16 patches → 768-dim
    │   ├─ 12 Transformer blocks (12 heads, window attn)
    │   ├─ Neck: Conv2d layers → 256-dim
    │   └─ Output: 1024-dim features [B, 1024, 64, 64]
    │
    └── CLIP-L (clip_sdpa.py)
        ├─ PatchEmbed: 14x14 patches → 1024-dim
        ├─ 24 Transformer blocks (16 heads)
        └─ Output: 1024-dim features [B, seq_len, 1024]
    ↓
Concatenate SAM + CLIP features → 2048-dim
    ↓
MLP Projector (build_linear.py)
    ├─ Linear projection: 2048 → 1280
    ├─ Optional downsampling (F.unfold)
    └─ Output: 1280-dim embeddings
    ↓
vLLM Engine (CUDA-only)
    ├─ DeepSeek-V2/V3 Language Model
    ├─ Multi-modal embedding merger
    └─ Autoregressive text generation
    ↓
Text Output (OCR results, markdown, etc.)
```

### Target Architecture (MLX/Metal)
```
Input Image (PIL)
    ↓
DeepseekOCRProcessor_MLX (image_process_mlx.py)
    ├─ Same preprocessing logic
    ├─ Output: mx.array instead of torch.Tensor
    └─ Output: [pixel_values, images_crop, images_spatial_crop]
    ↓
Vision Encoders (Parallel Processing)
    ├── SAM ViT-B MLX (sam_vary_mlx.py)
    │   ├─ mx.nn.Conv2d for patches
    │   ├─ mx.fast.scaled_dot_product_attention
    │   ├─ mx.nn.Conv2d for neck
    │   └─ Output: mx.array [B, 1024, 64, 64]
    │
    └── CLIP-L MLX (clip_mlx.py)
        ├─ mx.nn.Conv2d for patches
        ├─ mx.fast.scaled_dot_product_attention
        └─ Output: mx.array [B, seq_len, 1024]
    ↓
mx.concatenate SAM + CLIP → 2048-dim
    ↓
MLP Projector MLX (build_linear_mlx.py)
    ├─ mx.nn.Linear: 2048 → 1280
    ├─ unfold_mlx (reshape/transpose)
    └─ Output: 1280-dim mx.array
    ↓
MLX-LM Engine (Metal-optimized)
    ├─ MLX-compatible language model
    ├─ Multi-modal embedding merger (MLX)
    └─ Autoregressive generation
    ↓
Text Output (OCR results, markdown, etc.)
```

---

## Phase 1: Environment Setup & Validation

**Duration:** 0.5 days (4-6 hours)
**Prerequisites:** macOS with Apple Silicon (M1/M2/M3), Python 3.11+
**Deliverables:** Working MLX environment, validated installation, directory structure

### 1.1 Install MLX and Dependencies

**Checklist:**
- [ ] Create conda environment: `deepseek-mlx` (Python 3.11)
- [ ] Install MLX >=0.28.0
- [ ] Install MLX-LM >=0.10.0
- [ ] Install supporting packages (transformers, tokenizers, PIL, numpy)
- [ ] Verify MLX version and SDPA availability

**Commands:**
```bash
# Create environment
conda create -n deepseek-mlx python=3.11 -y
conda activate deepseek-mlx

# Install MLX packages
pip install mlx>=0.28.0 mlx-lm>=0.10.0

# Install dependencies
pip install transformers==4.46.3 tokenizers==0.20.3
pip install Pillow numpy easydict addict
pip install PyMuPDF img2pdf
pip install pytest pytest-benchmark pytest-cov

# Verify installation
python -c "import mlx.core as mx; print(f'MLX {mx.__version__}')"
python -c "import mlx.nn as nn; print('MLX NN OK')"
python -c "from mlx_lm import load; print('MLX-LM OK')"
```

**Validation Script:**
```python
# scripts/validate_mlx.py
import mlx.core as mx
import mlx.nn as nn

def validate_mlx_installation():
    """Validate MLX installation and key features"""
    print("=" * 60)
    print("MLX Installation Validation")
    print("=" * 60)

    # Check version
    print(f"\n1. MLX Version: {mx.__version__}")
    assert mx.__version__ >= "0.28.0", "MLX version must be >=0.28.0"

    # Test basic tensor operations
    print("\n2. Testing basic tensor operations...")
    x = mx.random.normal([2, 3, 224, 224])
    assert x.shape == [2, 3, 224, 224], "Tensor creation failed"
    print(f"   ✓ Created tensor: {x.shape}")

    # Test SDPA (critical for transformers)
    print("\n3. Testing Scaled Dot-Product Attention...")
    try:
        q = mx.random.normal([2, 8, 16, 64])
        k = mx.random.normal([2, 8, 16, 64])
        v = mx.random.normal([2, 8, 16, 64])
        out = mx.fast.scaled_dot_product_attention(q, k, v)
        assert out.shape == [2, 8, 16, 64], "SDPA output shape incorrect"
        print(f"   ✓ SDPA working: {out.shape}")
    except Exception as e:
        print(f"   ✗ SDPA failed: {e}")
        return False

    # Test Conv2d
    print("\n4. Testing Conv2d...")
    conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
    x = mx.random.normal([2, 3, 224, 224])
    out = conv(x)
    assert out.shape == [2, 64, 224, 224], "Conv2d failed"
    print(f"   ✓ Conv2d working: {out.shape}")

    # Test LayerNorm
    print("\n5. Testing LayerNorm...")
    ln = nn.LayerNorm(64)
    x = mx.random.normal([2, 64])
    out = ln(x)
    assert out.shape == [2, 64], "LayerNorm failed"
    print(f"   ✓ LayerNorm working: {out.shape}")

    # Test Linear
    print("\n6. Testing Linear...")
    linear = nn.Linear(128, 256)
    x = mx.random.normal([2, 128])
    out = linear(x)
    assert out.shape == [2, 256], "Linear failed"
    print(f"   ✓ Linear working: {out.shape}")

    # Test RMSNorm
    print("\n7. Testing RMSNorm...")
    rms = nn.RMSNorm(64)
    x = mx.random.normal([2, 64])
    out = rms(x)
    assert out.shape == [2, 64], "RMSNorm failed"
    print(f"   ✓ RMSNorm working: {out.shape}")

    # Test GELU
    print("\n8. Testing GELU activation...")
    gelu = nn.GELU()
    x = mx.random.normal([2, 64])
    out = gelu(x)
    assert out.shape == [2, 64], "GELU failed"
    print(f"   ✓ GELU working: {out.shape}")

    print("\n" + "=" * 60)
    print("✅ All validation checks passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = validate_mlx_installation()
    exit(0 if success else 1)
```

**Run validation:**
```bash
python scripts/validate_mlx.py
```

**Acceptance Criteria:**
- ✅ MLX version >= 0.28.0 installed
- ✅ `mx.fast.scaled_dot_product_attention` available and working
- ✅ All neural network layers (Conv2d, Linear, LayerNorm, RMSNorm, GELU) working
- ✅ No import errors or runtime errors
- ✅ Validation script passes all checks

---

### 1.2 Create Directory Structure

**Checklist:**
- [ ] Create `mlx_port/` directory
- [ ] Create subdirectories for encoders, process, tests, benchmarks
- [ ] Create `__init__.py` files
- [ ] Create requirements file
- [ ] Create initial configuration file

**Commands:**
```bash
cd /home/user/DeepSeek-OCR

# Create main directories
mkdir -p mlx_port/{deepencoder,process,scripts,tests,benchmarks}

# Create test subdirectories
mkdir -p mlx_port/tests/{unit,integration,validation}

# Create benchmark subdirectories
mkdir -p mlx_port/benchmarks/{outputs,results}

# Create __init__.py files
touch mlx_port/__init__.py
touch mlx_port/deepencoder/__init__.py
touch mlx_port/process/__init__.py
touch mlx_port/tests/__init__.py
touch mlx_port/tests/{unit,integration,validation}/__init__.py
touch mlx_port/scripts/__init__.py
```

**Directory Structure:**
```
mlx_port/
├── __init__.py
├── README.md                        # MLX port documentation
├── requirements_mlx.txt             # MLX dependencies
├── config_mlx.py                    # MLX-specific configuration
│
├── deepencoder/                     # Vision encoders
│   ├── __init__.py
│   ├── utils_mlx.py                 # MLX utility functions (unfold, interpolate, etc.)
│   ├── sam_vary_mlx.py              # SAM ViT-B encoder (MLX)
│   ├── clip_mlx.py                  # CLIP-L encoder (MLX)
│   └── build_linear_mlx.py          # MLP projector (MLX)
│
├── process/                         # Image processing
│   ├── __init__.py
│   ├── image_process_mlx.py         # Image processor (MLX)
│   └── ngram_norepeat.py            # N-gram no-repeat (copy from original)
│
├── deepseek_ocr_mlx.py              # Main MLX model
├── run_dpsk_ocr_mlx.py              # MLX inference script
│
├── scripts/                         # Utility scripts
│   ├── validate_mlx.py              # MLX installation validator
│   ├── convert_weights.py           # PyTorch → MLX weight converter
│   └── compare_outputs.py           # PyTorch vs MLX output comparator
│
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── conftest.py                  # pytest configuration
│   ├── test_utils.py                # Test utilities (comparators, fixtures)
│   │
│   ├── unit/                        # Unit tests
│   │   ├── test_utils_mlx.py        # Test utility functions
│   │   ├── test_sam_encoder.py      # Test SAM encoder
│   │   ├── test_clip_encoder.py     # Test CLIP encoder
│   │   └── test_projector.py        # Test MLP projector
│   │
│   ├── integration/                 # Integration tests
│   │   ├── test_vision_pipeline.py  # Test complete vision processing
│   │   └── test_full_model.py       # Test end-to-end model
│   │
│   └── validation/                  # Validation tests
│       ├── test_pytorch_parity.py   # Compare PyTorch vs MLX outputs
│       └── test_numerical_accuracy.py # Numerical accuracy tests
│
└── benchmarks/                      # Performance benchmarks
    ├── benchmark_inference.py       # Inference latency/throughput
    ├── benchmark_memory.py          # Memory usage profiling
    ├── compare_performance.py       # PyTorch vs MLX comparison
    ├── outputs/                     # Benchmark output files
    └── results/                     # Benchmark results (JSON, CSV)
```

**Create requirements file:**
```bash
cat > mlx_port/requirements_mlx.txt << 'EOF'
# MLX framework
mlx>=0.28.0
mlx-lm>=0.10.0

# Transformers ecosystem
transformers==4.46.3
tokenizers==0.20.3

# Image processing
Pillow>=10.0.0
numpy>=1.24.0

# Utilities
easydict>=1.10
addict>=2.4.0

# Document processing
PyMuPDF>=1.23.0
img2pdf>=0.5.0

# Testing
pytest>=7.4.0
pytest-benchmark>=4.0.0
pytest-cov>=4.1.0
pytest-xdist>=3.3.0  # Parallel testing

# Development
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0
EOF
```

**Create configuration file:**
```python
# mlx_port/config_mlx.py
"""MLX-specific configuration for DeepSeek-OCR"""

# Model paths
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'

# Image processing settings
BASE_SIZE = 1024          # Base image size
IMAGE_SIZE = 640          # Crop image size
CROP_MODE = True          # Enable dynamic cropping
MIN_CROPS = 2             # Minimum number of crops
MAX_CROPS = 6             # Maximum number of crops

# Inference settings
MAX_TOKENS = 8192         # Maximum generation length
TEMPERATURE = 0.0         # Sampling temperature (0.0 = greedy)

# Input/Output paths
INPUT_PATH = ''           # Set at runtime
OUTPUT_PATH = ''          # Set at runtime

# Prompt template
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'

# MLX-specific settings
MLX_DTYPE = 'bfloat16'    # Default dtype for MLX (bfloat16, float16, float32)
MLX_CACHE_LIMIT_GB = 8    # MLX cache limit in GB

# Performance settings
BATCH_SIZE = 1            # Batch size for inference (currently only supports 1)
NUM_WORKERS = 4           # Number of workers for data loading

# Testing settings
TEST_TOLERANCE_RTOL = 1e-4  # Relative tolerance for numerical comparisons
TEST_TOLERANCE_ATOL = 1e-5  # Absolute tolerance for numerical comparisons

# Logging
PRINT_NUM_VIS_TOKENS = False  # Print number of vision tokens
SKIP_REPEAT = True            # Skip n-gram repetition

# Tokenizer (lazy loaded)
from transformers import AutoTokenizer
_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return _tokenizer

TOKENIZER = property(lambda self: get_tokenizer())
```

**Acceptance Criteria:**
- ✅ All directories created
- ✅ `__init__.py` files in all packages
- ✅ `requirements_mlx.txt` created
- ✅ `config_mlx.py` created with proper settings
- ✅ Directory structure matches specification

---

### 1.3 Set Up Testing Framework

**Checklist:**
- [ ] Create pytest configuration
- [ ] Create test utilities module
- [ ] Create fixtures for common test data
- [ ] Create PyTorch-MLX comparator class
- [ ] Verify pytest works

**Create pytest configuration:**
```python
# mlx_port/tests/conftest.py
"""pytest configuration and fixtures"""
import pytest
import torch
import mlx.core as mx
import numpy as np
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def test_image_path():
    """Path to test image"""
    return Path(__file__).parent / "fixtures" / "test_image.jpg"


@pytest.fixture(scope="session")
def pytorch_model_path():
    """Path to PyTorch model weights"""
    return "deepseek-ai/DeepSeek-OCR"


@pytest.fixture
def random_image_tensor():
    """Random image tensor for testing"""
    # Return both PyTorch and MLX versions
    torch_tensor = torch.randn(2, 3, 224, 224)
    mlx_tensor = mx.array(torch_tensor.numpy())
    return torch_tensor, mlx_tensor


@pytest.fixture
def tolerance():
    """Default tolerance for numerical comparisons"""
    return {"rtol": 1e-4, "atol": 1e-5}


# pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "validation: marks tests as validation tests"
    )
```

**Create test utilities:**
```python
# mlx_port/tests/test_utils.py
"""Utilities for testing MLX implementation against PyTorch"""
import torch
import mlx.core as mx
import mlx.nn as nn_mlx
import torch.nn as nn_torch
import numpy as np
from typing import Dict, Any, Optional, Tuple


class PyTorchMLXComparator:
    """Utility class for comparing PyTorch and MLX outputs"""

    @staticmethod
    def torch_to_mlx(torch_tensor: torch.Tensor) -> mx.array:
        """
        Convert PyTorch tensor to MLX array.

        Args:
            torch_tensor: PyTorch tensor

        Returns:
            MLX array
        """
        return mx.array(torch_tensor.detach().cpu().numpy())

    @staticmethod
    def mlx_to_torch(mlx_array: mx.array) -> torch.Tensor:
        """
        Convert MLX array to PyTorch tensor.

        Args:
            mlx_array: MLX array

        Returns:
            PyTorch tensor
        """
        return torch.from_numpy(np.array(mlx_array))

    @staticmethod
    def assert_close(
        torch_out: torch.Tensor,
        mlx_out: mx.array,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        name: str = "",
        verbose: bool = True
    ):
        """
        Assert that PyTorch and MLX outputs are close.

        Args:
            torch_out: PyTorch output tensor
            mlx_out: MLX output array
            rtol: Relative tolerance
            atol: Absolute tolerance
            name: Name for logging
            verbose: Whether to print detailed comparison info

        Raises:
            AssertionError: If outputs differ beyond tolerance
        """
        # Convert to numpy
        torch_np = torch_out.detach().cpu().numpy()
        mlx_np = np.array(mlx_out)

        # Check shapes
        if torch_np.shape != mlx_np.shape:
            raise AssertionError(
                f"{name}: Shape mismatch - PyTorch: {torch_np.shape}, MLX: {mlx_np.shape}"
            )

        # Compute differences
        abs_diff = np.abs(torch_np - mlx_np)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)

        rel_diff = np.abs((torch_np - mlx_np) / (np.abs(torch_np) + 1e-8))
        max_rel_diff = np.max(rel_diff)
        mean_rel_diff = np.mean(rel_diff)

        if verbose:
            print(f"\n{'='*60}")
            print(f"{name} Comparison:")
            print(f"{'='*60}")
            print(f"  Shape: {torch_np.shape}")
            print(f"  Max absolute difference: {max_diff:.2e}")
            print(f"  Mean absolute difference: {mean_diff:.2e}")
            print(f"  Max relative difference: {max_rel_diff:.2e}")
            print(f"  Mean relative difference: {mean_rel_diff:.2e}")
            print(f"  Tolerance: rtol={rtol:.2e}, atol={atol:.2e}")

        # Assert closeness
        try:
            np.testing.assert_allclose(
                torch_np, mlx_np,
                rtol=rtol, atol=atol,
                err_msg=f"{name}: Outputs differ beyond tolerance"
            )
            if verbose:
                print(f"  ✅ PASSED (within tolerance)")
                print(f"{'='*60}")
        except AssertionError as e:
            if verbose:
                print(f"  ❌ FAILED (exceeds tolerance)")
                print(f"{'='*60}")
            raise e

    @staticmethod
    def load_pytorch_weights_to_mlx(
        mlx_module: nn_mlx.Module,
        pytorch_state_dict: Dict[str, torch.Tensor],
        prefix: str = "",
        strict: bool = True,
        verbose: bool = True
    ) -> nn_mlx.Module:
        """
        Load PyTorch weights into MLX module.

        Args:
            mlx_module: MLX module to load weights into
            pytorch_state_dict: PyTorch state dict
            prefix: Prefix to remove from keys
            strict: Whether to raise error on missing/unexpected keys
            verbose: Whether to print loading info

        Returns:
            MLX module with loaded weights
        """
        mlx_params = dict(mlx_module.parameters())
        loaded_keys = set()
        missing_keys = set()

        if verbose:
            print(f"\nLoading PyTorch weights into MLX module...")
            print(f"  PyTorch state dict keys: {len(pytorch_state_dict)}")
            print(f"  MLX parameter keys: {len(mlx_params)}")

        for key, value in pytorch_state_dict.items():
            # Remove prefix if specified
            mlx_key = key
            if prefix and key.startswith(prefix):
                mlx_key = key[len(prefix):]

            if mlx_key in mlx_params:
                # Convert PyTorch tensor to MLX array
                mlx_params[mlx_key] = mx.array(value.detach().cpu().numpy())
                loaded_keys.add(mlx_key)
            elif strict:
                if verbose:
                    print(f"  Warning: Key '{key}' not found in MLX module")

        # Check for missing keys
        for key in mlx_params:
            if key not in loaded_keys:
                missing_keys.add(key)

        if verbose:
            print(f"  Loaded keys: {len(loaded_keys)}")
            if missing_keys:
                print(f"  Missing keys: {len(missing_keys)}")
                for key in list(missing_keys)[:5]:
                    print(f"    - {key}")
                if len(missing_keys) > 5:
                    print(f"    ... and {len(missing_keys) - 5} more")

        if strict and missing_keys:
            raise ValueError(f"Missing keys in MLX module: {missing_keys}")

        # Update module parameters
        mlx_module.update(mlx_params)

        if verbose:
            print(f"  ✅ Weights loaded successfully")

        return mlx_module

    @staticmethod
    def compare_layer_outputs(
        torch_layer: nn_torch.Module,
        mlx_layer: nn_mlx.Module,
        input_tensor: torch.Tensor,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        name: str = "Layer"
    ) -> Tuple[torch.Tensor, mx.array]:
        """
        Compare outputs of PyTorch and MLX layers.

        Args:
            torch_layer: PyTorch layer
            mlx_layer: MLX layer
            input_tensor: Input tensor (PyTorch)
            rtol: Relative tolerance
            atol: Absolute tolerance
            name: Layer name for logging

        Returns:
            Tuple of (PyTorch output, MLX output)
        """
        # Convert input
        mlx_input = PyTorchMLXComparator.torch_to_mlx(input_tensor)

        # Forward pass
        with torch.no_grad():
            torch_output = torch_layer(input_tensor)

        mlx_output = mlx_layer(mlx_input)

        # Compare
        PyTorchMLXComparator.assert_close(
            torch_output, mlx_output,
            rtol=rtol, atol=atol,
            name=name
        )

        return torch_output, mlx_output


def create_test_image(batch_size: int = 1, size: int = 224) -> Tuple[torch.Tensor, mx.array]:
    """
    Create test image tensors for PyTorch and MLX.

    Args:
        batch_size: Batch size
        size: Image size (H and W)

    Returns:
        Tuple of (PyTorch tensor, MLX array)
    """
    torch_img = torch.randn(batch_size, 3, size, size)
    mlx_img = mx.array(torch_img.numpy())
    return torch_img, mlx_img
```

**Run initial tests:**
```bash
# Run pytest with verbose output
cd mlx_port
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

**Acceptance Criteria:**
- ✅ pytest configuration working
- ✅ Test utilities module created
- ✅ Fixtures defined and working
- ✅ PyTorchMLXComparator class functional
- ✅ Can convert between PyTorch and MLX tensors
- ✅ Can compare outputs with specified tolerances
- ✅ pytest runs without errors

---

## Phase 2: Core MLX Utilities Implementation

**Duration:** 0.5 days (4-6 hours)
**Prerequisites:** Phase 1 complete, MLX environment ready
**Deliverables:** `utils_mlx.py` with all helper functions, passing unit tests

### 2.1 Implement unfold_mlx()

**Location:** `mlx_port/deepencoder/utils_mlx.py`

**Technical Background:**

PyTorch's `F.unfold` extracts sliding local blocks from a batched tensor. In DeepSeek-OCR, it's **only used with stride == kernel_size** (non-overlapping patches), which makes the implementation trivial with reshape/transpose.

**Mathematical Background:**
```
Input: x ∈ ℝ^(B×C×H×W)
Operation: Extract k×k non-overlapping patches
Output: y ∈ ℝ^(B×(C·k²)×((H/k)·(W/k)))

Steps:
1. Reshape: [B, C, H, W] → [B, C, H/k, k, W/k, k]
2. Transpose: → [B, H/k, W/k, C, k, k]
3. Flatten patches: → [B, (H/k)·(W/k), C·k²]
4. Transpose to match PyTorch: → [B, C·k², (H/k)·(W/k)]
```

**Implementation:**

```python
# mlx_port/deepencoder/utils_mlx.py
"""MLX utility functions for DeepSeek-OCR"""
import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Union, List, Optional


def unfold_mlx(
    x: mx.array,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    padding: int = 0
) -> mx.array:
    """
    MLX implementation of F.unfold for non-overlapping patches.

    This function extracts sliding local blocks from a batched input tensor.
    **Only supports non-overlapping patches (stride == kernel_size)**.

    Args:
        x: Input array of shape [B, C, H, W]
        kernel_size: Size of the sliding blocks (int or tuple of 2 ints)
        stride: Stride of the sliding blocks (int or tuple of 2 ints)
        padding: Implicit zero padding (must be 0, not implemented)

    Returns:
        Array of shape [B, C*kernel_h*kernel_w, num_patches]
        where num_patches = (H/kernel_h) * (W/kernel_w)

    Raises:
        NotImplementedError: If padding != 0 or stride != kernel_size
        ValueError: If dimensions not divisible by kernel_size

    Example:
        >>> x = mx.random.normal([2, 3, 4, 4])
        >>> out = unfold_mlx(x, kernel_size=2, stride=2)
        >>> out.shape
        [2, 12, 4]  # 12 = 3*2*2, 4 = (4/2)*(4/2)
    """
    if padding != 0:
        raise NotImplementedError("unfold_mlx only supports padding=0")

    b, c, h, w = x.shape

    # Handle tuple or int kernel_size
    if isinstance(kernel_size, int):
        kh, kw = kernel_size, kernel_size
    else:
        kh, kw = kernel_size

    # Handle tuple or int stride
    if isinstance(stride, int):
        sh, sw = stride, stride
    else:
        sh, sw = stride

    # Verify non-overlapping (this is the simplification that makes it trivial)
    if sh != kh or sw != kw:
        raise NotImplementedError(
            f"unfold_mlx only supports non-overlapping patches (stride == kernel_size). "
            f"Got stride=({sh}, {sw}), kernel_size=({kh}, {kw})"
        )

    # Ensure dimensions are divisible
    if h % kh != 0 or w % kw != 0:
        raise ValueError(
            f"Height ({h}) and width ({w}) must be divisible by "
            f"kernel_size ({kh}, {kw})"
        )

    # Step 1: Reshape to separate patches
    # [B, C, H, W] → [B, C, H//kh, kh, W//kw, kw]
    x = x.reshape([b, c, h // kh, kh, w // kw, kw])

    # Step 2: Rearrange dimensions to group spatial patches
    # [B, C, H//kh, kh, W//kw, kw] → [B, H//kh, W//kw, C, kh, kw]
    x = x.transpose([0, 2, 4, 1, 3, 5])

    # Step 3: Flatten patches
    # [B, H//kh, W//kw, C, kh, kw] → [B, (H//kh)*(W//kw), C*kh*kw]
    num_patches_h = h // kh
    num_patches_w = w // kw
    x = x.reshape([b, num_patches_h * num_patches_w, c * kh * kw])

    # Step 4: Transpose to match PyTorch F.unfold output format
    # [B, num_patches, C*kh*kw] → [B, C*kh*kw, num_patches]
    x = x.transpose([0, 2, 1])

    return x


def interpolate_mlx(
    x: mx.array,
    size: Union[Tuple[int, int], List[int]],
    mode: str = 'bicubic',
    align_corners: bool = False,
    antialias: bool = True
) -> mx.array:
    """
    MLX implementation of F.interpolate for image resizing.

    Args:
        x: Input array of shape [B, C, H, W]
        size: Target size as (H, W)
        mode: Interpolation mode ('bilinear', 'bicubic', 'linear')
        align_corners: Whether to align corners (currently not used in MLX)
        antialias: Whether to use antialiasing

    Returns:
        Resized array of shape [B, C, size[0], size[1]]

    Example:
        >>> x = mx.random.normal([2, 3, 64, 64])
        >>> out = interpolate_mlx(x, (32, 32), mode='bicubic')
        >>> out.shape
        [2, 3, 32, 32]
    """
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise ValueError(f"size must be a tuple/list of 2 elements, got {size}")

    b, c, h, w = x.shape
    target_h, target_w = size

    # If already target size, return as-is
    if h == target_h and w == target_w:
        return x

    # MLX's image.resize operates on [H, W, C] format
    # We need to handle batch and channel dimensions

    # Approach: Process each image in batch separately
    resized_images = []

    for i in range(b):
        # Get single image: [C, H, W]
        img = x[i]

        # Transpose to [H, W, C] for mx.image.resize
        img = img.transpose([1, 2, 0])

        # Resize using MLX's image resize
        # Note: mx.image.resize expects size as [H, W]
        img_resized = mx.image.resize(
            img,
            [target_h, target_w],
            method=mode,
            antialias=antialias
        )

        # Transpose back to [C, H, W]
        img_resized = img_resized.transpose([2, 0, 1])

        resized_images.append(img_resized)

    # Stack batch: [B, C, H, W]
    result = mx.stack(resized_images, axis=0)

    return result


def pad_mlx(
    x: mx.array,
    pad: Tuple[int, ...],
    mode: str = 'constant',
    value: float = 0
) -> mx.array:
    """
    MLX implementation of F.pad.

    Args:
        x: Input array
        pad: Padding specification in PyTorch format (left, right, top, bottom, ...)
        mode: Padding mode ('constant', 'replicate'/'edge', 'reflect')
        value: Fill value for constant padding

    Returns:
        Padded array

    Note:
        PyTorch pad format: (left, right, top, bottom, front, back)
        MLX pad format: [(before, after), ...] for each dimension

    Example:
        >>> x = mx.random.normal([2, 3, 10, 10])
        >>> out = pad_mlx(x, (1, 1, 2, 2), mode='constant', value=0)
        >>> out.shape
        [2, 3, 12, 12]  # 10+2+2=14 height, 10+1+1=12 width
    """
    ndim = len(x.shape)

    # Convert PyTorch pad format to MLX format
    # PyTorch pad applies to last dimensions first
    mlx_pad = [(0, 0)] * ndim

    # Process pad values in pairs (before, after) for each dimension
    for i in range(len(pad) // 2):
        dim_idx = ndim - 1 - i  # Start from last dimension
        mlx_pad[dim_idx] = (pad[2 * i], pad[2 * i + 1])

    # Apply padding based on mode
    if mode == 'constant':
        return mx.pad(x, mlx_pad, constant_values=value)
    elif mode in ('replicate', 'edge'):
        return mx.pad(x, mlx_pad, mode='edge')
    elif mode == 'reflect':
        return mx.pad(x, mlx_pad, mode='reflect')
    else:
        raise ValueError(f"Unsupported padding mode: {mode}")


def get_abs_pos_mlx(abs_pos: mx.array, tgt_size: int) -> mx.array:
    """
    Resize absolute position embeddings to target size.

    Handles both CLIP-style (sequence with CLS token) and SAM-style (2D spatial)
    position embeddings.

    Args:
        abs_pos: Position embeddings
                 - CLIP style: [1, src_size²+1, C] (includes CLS token)
                 - SAM style: [1, src_size, src_size, C]
        tgt_size: Target spatial size

    Returns:
        Resized position embeddings in same format as input

    Example:
        >>> # CLIP-style
        >>> pos_embed = mx.random.normal([1, 257, 1024])  # 16x16 + 1 CLS
        >>> resized = get_abs_pos_mlx(pos_embed, 32)  # Resize to 32x32
        >>> resized.shape
        [1, 1025, 1024]  # 32x32 + 1 CLS

        >>> # SAM-style
        >>> pos_embed = mx.random.normal([1, 64, 64, 768])
        >>> resized = get_abs_pos_mlx(pos_embed, 32)
        >>> resized.shape
        [1, 32, 32, 768]
    """
    dtype = abs_pos.dtype

    # Handle CLIP-style: [1, seq_len, C] where seq_len = spatial_size² + 1
    if len(abs_pos.shape) == 3:
        # Calculate source spatial size (excluding CLS token)
        src_size = int((abs_pos.shape[1] - 1) ** 0.5)

        # Split CLS token and spatial embeddings
        cls_token = abs_pos[:, :1, :]  # [1, 1, C]
        pos_embed = abs_pos[:, 1:, :]   # [1, src_size², C]

        if src_size != tgt_size:
            # Reshape to 2D spatial: [1, src_size², C] → [1, src_size, src_size, C]
            c = pos_embed.shape[-1]
            pos_embed = pos_embed.reshape([1, src_size, src_size, c])

            # Transpose to [1, C, H, W] for resizing
            pos_embed = pos_embed.transpose([0, 3, 1, 2])

            # Resize
            pos_embed = pos_embed.astype(mx.float32)
            pos_embed = interpolate_mlx(
                pos_embed,
                (tgt_size, tgt_size),
                mode='bicubic',
                antialias=True
            )
            pos_embed = pos_embed.astype(dtype)

            # Transpose back and reshape: [1, C, H, W] → [1, tgt_size², C]
            pos_embed = pos_embed.transpose([0, 2, 3, 1])
            pos_embed = pos_embed.reshape([1, tgt_size * tgt_size, c])

            # Concatenate CLS token back
            return mx.concatenate([cls_token, pos_embed], axis=1)
        else:
            return abs_pos

    # Handle SAM-style: [1, H, W, C]
    elif len(abs_pos.shape) == 4:
        src_size = abs_pos.shape[1]

        if src_size != tgt_size:
            # Transpose to [1, C, H, W]
            abs_pos = abs_pos.transpose([0, 3, 1, 2])

            # Resize
            abs_pos = abs_pos.astype(mx.float32)
            abs_pos = interpolate_mlx(
                abs_pos,
                (tgt_size, tgt_size),
                mode='bicubic',
                antialias=True
            )
            abs_pos = abs_pos.astype(dtype)

            # Transpose back to [1, H, W, C]
            abs_pos = abs_pos.transpose([0, 2, 3, 1])

            return abs_pos
        else:
            return abs_pos
    else:
        raise ValueError(
            f"Unexpected abs_pos shape: {abs_pos.shape}. "
            f"Expected [1, seq_len, C] or [1, H, W, C]"
        )


def quick_gelu_mlx(x: mx.array) -> mx.array:
    """
    Quick GELU activation: x * sigmoid(1.702 * x)

    Used in CLIP encoder as a faster approximation of GELU.

    Args:
        x: Input array

    Returns:
        Activated array

    Example:
        >>> x = mx.random.normal([2, 64])
        >>> out = quick_gelu_mlx(x)
        >>> out.shape
        [2, 64]
    """
    return x * mx.sigmoid(1.702 * x)
```

**Acceptance Criteria:**
- ✅ `unfold_mlx()` implemented and documented
- ✅ `interpolate_mlx()` implemented and documented
- ✅ `pad_mlx()` implemented and documented
- ✅ `get_abs_pos_mlx()` implemented and documented
- ✅ `quick_gelu_mlx()` implemented and documented
- ✅ All functions have proper docstrings with examples
- ✅ Type hints provided for all functions

---

### 2.2 Unit Tests for Utilities

**Location:** `mlx_port/tests/unit/test_utils_mlx.py`

**Implementation:**

```python
# mlx_port/tests/unit/test_utils_mlx.py
"""Unit tests for MLX utility functions"""
import pytest
import torch
import torch.nn.functional as F
import mlx.core as mx
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deepencoder.utils_mlx import (
    unfold_mlx,
    interpolate_mlx,
    pad_mlx,
    get_abs_pos_mlx,
    quick_gelu_mlx
)


class TestUnfoldMLX:
    """Tests for unfold_mlx function"""

    def test_unfold_2x2_kernel(self):
        """Test unfold with 2x2 kernel"""
        # Create test input
        torch_input = torch.randn(2, 3, 4, 4)
        mlx_input = mx.array(torch_input.numpy())

        # PyTorch unfold
        torch_output = F.unfold(torch_input, kernel_size=2, stride=2, padding=0)

        # MLX unfold
        mlx_output = unfold_mlx(mlx_input, kernel_size=2, stride=2, padding=0)

        # Compare
        np.testing.assert_allclose(
            torch_output.numpy(),
            np.array(mlx_output),
            rtol=1e-6, atol=1e-6
        )

    def test_unfold_4x4_kernel(self):
        """Test unfold with 4x4 kernel"""
        torch_input = torch.randn(1, 64, 8, 8)
        mlx_input = mx.array(torch_input.numpy())

        torch_output = F.unfold(torch_input, kernel_size=4, stride=4, padding=0)
        mlx_output = unfold_mlx(mlx_input, kernel_size=4, stride=4, padding=0)

        np.testing.assert_allclose(
            torch_output.numpy(),
            np.array(mlx_output),
            rtol=1e-6, atol=1e-6
        )

    def test_unfold_tuple_kernel(self):
        """Test unfold with tuple kernel_size"""
        torch_input = torch.randn(1, 3, 8, 12)
        mlx_input = mx.array(torch_input.numpy())

        torch_output = F.unfold(torch_input, kernel_size=(2, 3), stride=(2, 3), padding=0)
        mlx_output = unfold_mlx(mlx_input, kernel_size=(2, 3), stride=(2, 3), padding=0)

        np.testing.assert_allclose(
            torch_output.numpy(),
            np.array(mlx_output),
            rtol=1e-6, atol=1e-6
        )

    def test_unfold_raises_on_overlapping(self):
        """Test that unfold raises error for overlapping patches"""
        mlx_input = mx.random.normal([1, 3, 8, 8])

        with pytest.raises(NotImplementedError, match="non-overlapping"):
            unfold_mlx(mlx_input, kernel_size=3, stride=2, padding=0)

    def test_unfold_raises_on_padding(self):
        """Test that unfold raises error for padding"""
        mlx_input = mx.random.normal([1, 3, 8, 8])

        with pytest.raises(NotImplementedError, match="padding=0"):
            unfold_mlx(mlx_input, kernel_size=2, stride=2, padding=1)


class TestInterpolateMLX:
    """Tests for interpolate_mlx function"""

    def test_interpolate_bicubic_downsample(self):
        """Test bicubic interpolation (downsampling)"""
        torch_input = torch.randn(2, 3, 64, 64)
        mlx_input = mx.array(torch_input.numpy())

        torch_output = F.interpolate(
            torch_input,
            size=(32, 32),
            mode='bicubic',
            align_corners=False,
            antialias=True
        )
        mlx_output = interpolate_mlx(mlx_input, size=(32, 32), mode='bicubic')

        # Allow larger tolerance for interpolation
        np.testing.assert_allclose(
            torch_output.numpy(),
            np.array(mlx_output),
            rtol=1e-3, atol=1e-4
        )

    def test_interpolate_bilinear(self):
        """Test bilinear interpolation"""
        torch_input = torch.randn(1, 3, 32, 32)
        mlx_input = mx.array(torch_input.numpy())

        torch_output = F.interpolate(
            torch_input,
            size=(64, 64),
            mode='bilinear',
            align_corners=False
        )
        mlx_output = interpolate_mlx(mlx_input, size=(64, 64), mode='bilinear')

        np.testing.assert_allclose(
            torch_output.numpy(),
            np.array(mlx_output),
            rtol=1e-3, atol=1e-4
        )

    def test_interpolate_no_resize(self):
        """Test that no resizing occurs when size matches"""
        mlx_input = mx.random.normal([2, 3, 32, 32])
        mlx_output = interpolate_mlx(mlx_input, size=(32, 32))

        # Should be identical (no operation)
        np.testing.assert_array_equal(
            np.array(mlx_input),
            np.array(mlx_output)
        )


class TestPadMLX:
    """Tests for pad_mlx function"""

    def test_pad_constant(self):
        """Test constant padding"""
        torch_input = torch.randn(2, 3, 10, 10)
        mlx_input = mx.array(torch_input.numpy())

        pad_spec = (1, 1, 2, 2)  # left, right, top, bottom
        torch_output = F.pad(torch_input, pad_spec, mode='constant', value=0)
        mlx_output = pad_mlx(mlx_input, pad_spec, mode='constant', value=0)

        np.testing.assert_allclose(
            torch_output.numpy(),
            np.array(mlx_output),
            rtol=1e-6, atol=1e-6
        )

    def test_pad_reflect(self):
        """Test reflect padding"""
        torch_input = torch.randn(1, 3, 8, 8)
        mlx_input = mx.array(torch_input.numpy())

        pad_spec = (1, 1, 1, 1)
        torch_output = F.pad(torch_input, pad_spec, mode='reflect')
        mlx_output = pad_mlx(mlx_input, pad_spec, mode='reflect')

        np.testing.assert_allclose(
            torch_output.numpy(),
            np.array(mlx_output),
            rtol=1e-6, atol=1e-6
        )

    def test_pad_replicate(self):
        """Test replicate/edge padding"""
        torch_input = torch.randn(1, 3, 6, 6)
        mlx_input = mx.array(torch_input.numpy())

        pad_spec = (2, 2, 2, 2)
        torch_output = F.pad(torch_input, pad_spec, mode='replicate')
        mlx_output = pad_mlx(mlx_input, pad_spec, mode='edge')  # 'edge' is MLX equivalent

        np.testing.assert_allclose(
            torch_output.numpy(),
            np.array(mlx_output),
            rtol=1e-6, atol=1e-6
        )


class TestGetAbsPosMLX:
    """Tests for get_abs_pos_mlx function"""

    def test_get_abs_pos_clip_style_resize(self):
        """Test resizing CLIP-style position embeddings"""
        # CLIP style: [1, 257, 1024] = 16x16 spatial + 1 CLS token
        src_size = 16
        tgt_size = 32
        embed_dim = 1024

        abs_pos = mx.random.normal([1, src_size * src_size + 1, embed_dim])
        resized = get_abs_pos_mlx(abs_pos, tgt_size)

        # Check shape
        expected_shape = [1, tgt_size * tgt_size + 1, embed_dim]
        assert list(resized.shape) == expected_shape

    def test_get_abs_pos_clip_style_no_resize(self):
        """Test CLIP-style with no resizing needed"""
        src_size = 16
        embed_dim = 768

        abs_pos = mx.random.normal([1, src_size * src_size + 1, embed_dim])
        resized = get_abs_pos_mlx(abs_pos, src_size)

        # Should be identical
        np.testing.assert_array_equal(
            np.array(abs_pos),
            np.array(resized)
        )

    def test_get_abs_pos_sam_style_resize(self):
        """Test resizing SAM-style position embeddings"""
        # SAM style: [1, 64, 64, 768]
        src_size = 64
        tgt_size = 32
        embed_dim = 768

        abs_pos = mx.random.normal([1, src_size, src_size, embed_dim])
        resized = get_abs_pos_mlx(abs_pos, tgt_size)

        # Check shape
        expected_shape = [1, tgt_size, tgt_size, embed_dim]
        assert list(resized.shape) == expected_shape

    def test_get_abs_pos_sam_style_no_resize(self):
        """Test SAM-style with no resizing needed"""
        src_size = 32
        embed_dim = 256

        abs_pos = mx.random.normal([1, src_size, src_size, embed_dim])
        resized = get_abs_pos_mlx(abs_pos, src_size)

        # Should be identical
        np.testing.assert_array_equal(
            np.array(abs_pos),
            np.array(resized)
        )


class TestQuickGELUMLX:
    """Tests for quick_gelu_mlx function"""

    def test_quick_gelu_shape(self):
        """Test that quick_gelu preserves shape"""
        x = mx.random.normal([2, 64])
        out = quick_gelu_mlx(x)

        assert out.shape == x.shape

    def test_quick_gelu_values(self):
        """Test quick_gelu computation"""
        x = mx.array([0.0, 1.0, -1.0, 2.0])
        out = quick_gelu_mlx(x)

        # Compute expected: x * sigmoid(1.702 * x)
        expected = x * mx.sigmoid(1.702 * x)

        np.testing.assert_allclose(
            np.array(out),
            np.array(expected),
            rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])
```

**Run tests:**
```bash
cd mlx_port
pytest tests/unit/test_utils_mlx.py -v
```

**Acceptance Criteria:**
- ✅ All unfold_mlx tests pass (4+ test cases)
- ✅ All interpolate_mlx tests pass (3+ test cases)
- ✅ All pad_mlx tests pass (3+ test cases)
- ✅ All get_abs_pos_mlx tests pass (4+ test cases)
- ✅ All quick_gelu_mlx tests pass (2+ test cases)
- ✅ Test coverage >90% for utils_mlx.py
- ✅ No numerical differences beyond tolerance (rtol=1e-6 for exact ops, rtol=1e-3 for interpolation)

---

## Phase 3: SAM Vision Encoder Migration

**Duration:** 1.5 days (10-12 hours)
**Prerequisites:** Phase 2 complete, utilities tested
**Deliverables:** `sam_vary_mlx.py` with complete SAM encoder, passing unit tests

### 3.1 Implement Core Attention Mechanism

**Location:** `mlx_port/deepencoder/sam_vary_mlx.py`

**Key Technical Points:**

1. **Attention with Relative Position Bias:**
   - SAM uses relative position embeddings added as attention bias
   - Decomposed into height and width components
   - Requires careful einsum operations

2. **Window Attention:**
   - Some blocks use window attention (14x14 windows)
   - Others use global attention
   - Window partition/unpartition logic needed

3. **Flash Attention Replacement:**
   - Replace manual attention computation with `mx.fast.scaled_dot_product_attention`
   - Handle attention bias/mask properly

**Implementation:**

```python
# mlx_port/deepencoder/sam_vary_mlx.py
"""SAM ViT-B Vision Encoder (MLX Implementation)"""
import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, Type
from functools import partial

from .utils_mlx import get_abs_pos_mlx, interpolate_mlx


class MLPBlock(nn.Module):
    """MLP block with GELU activation"""

    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def __call__(self, x: mx.array) -> mx.array:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    """2D Layer Normalization for image features"""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones([num_channels])
        self.bias = mx.zeros([num_channels])
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, C, H, W]
        u = mx.mean(x, axis=1, keepdims=True)
        s = mx.mean((x - u) ** 2, axis=1, keepdims=True)
        x = (x - u) / mx.sqrt(s + self.eps)

        # Reshape weight and bias for broadcasting
        weight = self.weight.reshape([1, -1, 1, 1])
        bias = self.bias.reshape([1, -1, 1, 1])

        return weight * x + bias


class Attention(nn.Module):
    """Multi-head Attention with optional relative position embeddings"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projection
            use_rel_pos: Whether to use relative positional embeddings
            rel_pos_zero_init: Whether to zero-initialize rel pos embeddings
            input_size: Input spatial size (H, W) if using relative position
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, \
                "Input size must be provided if using relative positional encoding."

            # Initialize relative positional embeddings
            # [2*H-1, head_dim], [2*W-1, head_dim]
            if rel_pos_zero_init:
                self.rel_pos_h = mx.zeros([2 * input_size[0] - 1, head_dim])
                self.rel_pos_w = mx.zeros([2 * input_size[1] - 1, head_dim])
            else:
                self.rel_pos_h = mx.random.normal([2 * input_size[0] - 1, head_dim]) * 0.02
                self.rel_pos_w = mx.random.normal([2 * input_size[1] - 1, head_dim]) * 0.02

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Input tensor of shape [B, H, W, C]

        Returns:
            Output tensor of shape [B, H, W, C]
        """
        B, H, W, _ = x.shape

        # QKV projection: [B, H, W, C] → [B, H*W, 3, num_heads, head_dim]
        qkv = self.qkv(x).reshape([B, H * W, 3, self.num_heads, -1])

        # Split into q, k, v and transpose to [B, num_heads, H*W, head_dim]
        q, k, v = mx.split(qkv, 3, axis=2)
        q = mx.squeeze(q, axis=2).transpose([0, 2, 1, 3])  # [B, num_heads, H*W, head_dim]
        k = mx.squeeze(k, axis=2).transpose([0, 2, 1, 3])
        v = mx.squeeze(v, axis=2).transpose([0, 2, 1, 3])

        # Apply attention
        if self.use_rel_pos:
            # Compute relative position bias
            rel_h, rel_w = self._add_decomposed_rel_pos(q, (H, W), (H, W))

            # Combine height and width biases: [B, num_heads, H*W, H*W]
            attn_bias = (rel_h + rel_w.transpose([0, 1, 2, 4, 3])).reshape(
                [B, self.num_heads, H * W, H * W]
            )

            # Scaled dot-product attention with bias
            x = mx.fast.scaled_dot_product_attention(q, k, v, mask=attn_bias)
        else:
            # Scaled dot-product attention without bias
            x = mx.fast.scaled_dot_product_attention(q, k, v)

        # Reshape output: [B, num_heads, H*W, head_dim] → [B, H, W, C]
        x = x.transpose([0, 2, 1, 3]).reshape([B, H, W, -1])

        # Output projection
        x = self.proj(x)

        return x

    def _add_decomposed_rel_pos(
        self,
        q: mx.array,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
    ) -> Tuple[mx.array, mx.array]:
        """
        Calculate decomposed Relative Positional Embeddings.

        Args:
            q: Query tensor [B, num_heads, H*W, head_dim]
            q_size: Query spatial size (H, W)
            k_size: Key spatial size (H, W)

        Returns:
            Tuple of (rel_h, rel_w) bias tensors
        """
        q_h, q_w = q_size
        k_h, k_w = k_size

        # Get relative position embeddings
        Rh = self._get_rel_pos(q_h, k_h, self.rel_pos_h)  # [q_h, k_h, head_dim]
        Rw = self._get_rel_pos(q_w, k_w, self.rel_pos_w)  # [q_w, k_w, head_dim]

        B, num_heads, _, dim = q.shape

        # Reshape q: [B, num_heads, H*W, head_dim] → [B*num_heads, H, W, head_dim]
        r_q = q.reshape([B * num_heads, q_h, q_w, dim])

        # Compute height contribution using einsum
        # [B*num_heads, H, W, C] @ [H, k_h, C] → [B*num_heads, H, W, k_h]
        rel_h = mx.einsum('bhwc,hkc->bhwk', r_q, Rh)

        # Compute width contribution
        # [B*num_heads, H, W, C] @ [W, k_w, C] → [B*num_heads, H, W, k_w]
        rel_w = mx.einsum('bhwc,wkc->bhwk', r_q, Rw)

        # Reshape for broadcasting:
        # rel_h: [B*num_heads, H*W, k_h, 1]
        # rel_w: [B*num_heads, H*W, 1, k_w]
        rel_h = rel_h.reshape([B * num_heads, q_h * q_w, k_h, 1])
        rel_w = rel_w.reshape([B * num_heads, q_h * q_w, 1, k_w])

        # Reshape back to separate batch and heads
        rel_h = rel_h.reshape([B, num_heads, q_h * q_w, k_h, 1])
        rel_w = rel_w.reshape([B, num_heads, q_h * q_w, 1, k_w])

        return rel_h, rel_w

    @staticmethod
    def _get_rel_pos(q_size: int, k_size: int, rel_pos: mx.array) -> mx.array:
        """
        Get relative positional embeddings according to query and key sizes.

        Args:
            q_size: Size of query
            k_size: Size of key
            rel_pos: Relative position embeddings [L, C]

        Returns:
            Extracted position embeddings [q_size, k_size, C]
        """
        max_rel_dist = int(2 * max(q_size, k_size) - 1)

        # Interpolate if needed
        if rel_pos.shape[0] != max_rel_dist:
            dtype = rel_pos.dtype
            rel_pos = rel_pos.astype(mx.float32)

            # Reshape for interpolation: [L, C] → [1, C, L, 1]
            rel_pos_resized = rel_pos.transpose([1, 0]).reshape([1, -1, rel_pos.shape[0], 1])

            # Interpolate to max_rel_dist
            rel_pos_resized = interpolate_mlx(
                rel_pos_resized,
                (max_rel_dist, 1),
                mode='linear'
            )

            # Reshape back: [1, C, max_rel_dist, 1] → [max_rel_dist, C]
            rel_pos_resized = rel_pos_resized.reshape([-1, max_rel_dist]).transpose([1, 0])
            rel_pos_resized = rel_pos_resized.astype(dtype)
        else:
            rel_pos_resized = rel_pos

        # Scale coordinates with short length if shapes for q and k are different
        q_coords = mx.arange(q_size).reshape([-1, 1]) * max(k_size / q_size, 1.0)
        k_coords = mx.arange(k_size).reshape([1, -1]) * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

        return rel_pos_resized[relative_coords.astype(mx.int32)]


def window_partition(x: mx.array, window_size: int) -> Tuple[mx.array, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.

    Args:
        x: Input tensor [B, H, W, C]
        window_size: Window size

    Returns:
        windows: [B*num_windows, window_size, window_size, C]
        (Hp, Wp): Padded height and width before partition
    """
    B, H, W, C = x.shape

    # Calculate padding
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    # Apply padding if needed
    if pad_h > 0 or pad_w > 0:
        # Pad: [B, H, W, C] → [B, H+pad_h, W+pad_w, C]
        x = mx.pad(x, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)])

    Hp, Wp = H + pad_h, W + pad_w

    # Reshape to windows
    # [B, Hp, Wp, C] → [B, Hp//ws, ws, Wp//ws, ws, C]
    x = x.reshape([B, Hp // window_size, window_size, Wp // window_size, window_size, C])

    # Permute: [B, Hp//ws, Wp//ws, ws, ws, C]
    windows = x.transpose([0, 1, 3, 2, 4, 5])

    # Flatten: [B*num_windows, ws, ws, C]
    windows = windows.reshape([-1, window_size, window_size, C])

    return windows, (Hp, Wp)


def window_unpartition(
    windows: mx.array,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int]
) -> mx.array:
    """
    Window unpartition into original sequences and removing padding.

    Args:
        windows: [B*num_windows, window_size, window_size, C]
        window_size: Window size
        pad_hw: Padded (H, W)
        hw: Original (H, W) before padding

    Returns:
        x: [B, H, W, C]
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    C = windows.shape[-1]

    # Reshape: [B*num_windows, ws, ws, C] → [B, Hp//ws, Wp//ws, ws, ws, C]
    x = windows.reshape([B, Hp // window_size, Wp // window_size, window_size, window_size, C])

    # Permute: [B, Hp//ws, ws, Wp//ws, ws, C]
    x = x.transpose([0, 1, 3, 2, 4, 5])

    # Reshape: [B, Hp, Wp, C]
    x = x.reshape([B, Hp, Wp, C])

    # Remove padding if needed
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]

    return x


class Block(nn.Module):
    """Transformer block with support of window attention and residual propagation"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Args:
            dim: Number of input channels
            num_heads: Number of attention heads
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            qkv_bias: Whether to add bias in QKV projection
            norm_layer: Normalization layer
            act_layer: Activation layer
            use_rel_pos: Whether to use relative position embeddings
            rel_pos_zero_init: Whether to zero-initialize relative position
            window_size: Window size for window attention (0 = global)
            input_size: Input resolution for relative position
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(
            embedding_dim=dim,
            mlp_dim=int(dim * mlp_ratio),
            act=act_layer
        )

        self.window_size = window_size

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [B, H, W, C]

        Returns:
            [B, H, W, C]
        """
        shortcut = x
        x = self.norm1(x)

        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        # Attention
        x = self.attn(x)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        # Residual connection
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding using Conv2d"""

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        """
        Args:
            kernel_size: Patch size
            stride: Patch stride
            padding: Padding
            in_chans: Number of input channels
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [B, C, H, W]

        Returns:
            [B, H', W', embed_dim] where H'=H/stride, W'=W/stride
        """
        # Conv2d: [B, C, H, W] → [B, embed_dim, H', W']
        x = self.proj(x)

        # Permute: [B, embed_dim, H', W'] → [B, H', W', embed_dim]
        x = x.transpose([0, 2, 3, 1])

        return x


class ImageEncoderViT(nn.Module):
    """Vision Transformer encoder from SAM"""

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ):
        """
        Args:
            img_size: Input image size
            patch_size: Patch size
            in_chans: Number of input image channels
            embed_dim: Patch embedding dimension
            depth: Depth of ViT
            num_heads: Number of attention heads
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            out_chans: Output channels
            qkv_bias: Whether to use bias in QKV projection
            norm_layer: Normalization layer
            act_layer: Activation layer
            use_abs_pos: Whether to use absolute positional embeddings
            use_rel_pos: Whether to use relative positional embeddings
            rel_pos_zero_init: Whether to zero-initialize relative position
            window_size: Window size for window attention blocks
            global_attn_indexes: Indexes for blocks using global attention
        """
        super().__init__()
        self.img_size = img_size

        # Patch embedding
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # Absolute positional embedding
        self.pos_embed = None
        if use_abs_pos:
            self.pos_embed = mx.zeros([1, img_size // patch_size, img_size // patch_size, embed_dim])

        # Transformer blocks
        self.blocks = []
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        # Neck (output projection)
        self.neck = [
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        ]

        # Additional conv layers for deeper features
        self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [B, 3, H, W] input image

        Returns:
            [B, 1024, H/64, W/64] encoded features
        """
        # Patch embedding: [B, 3, H, W] → [B, H/16, W/16, embed_dim]
        x = self.patch_embed(x)

        # Add absolute position embedding if enabled
        if self.pos_embed is not None:
            x = x + get_abs_pos_mlx(self.pos_embed, x.shape[1])

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # Neck: [B, H/16, W/16, embed_dim] → [B, embed_dim, H/16, W/16]
        x = x.transpose([0, 3, 1, 2])

        # Apply neck layers sequentially
        for layer in self.neck:
            x = layer(x)

        # Apply additional conv layers
        # [B, 256, H/16, W/16] → [B, 512, H/32, W/32]
        conv2_output = self.net_2(x)

        # [B, 512, H/32, W/32] → [B, 1024, H/64, W/64]
        conv3_output = self.net_3(conv2_output)

        return conv3_output


def build_sam_vit_b(checkpoint: Optional[str] = None) -> ImageEncoderViT:
    """
    Build SAM ViT-B vision encoder.

    Args:
        checkpoint: Path to checkpoint file (not used in MLX, weights loaded separately)

    Returns:
        ImageEncoderViT model
    """
    return ImageEncoderViT(
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        out_chans=256,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_abs_pos=True,
        use_rel_pos=True,
        rel_pos_zero_init=True,
        window_size=14,
        global_attn_indexes=[2, 5, 8, 11],
    )
```

**Acceptance Criteria:**
- ✅ All SAM encoder classes implemented
- ✅ Attention mechanism uses `mx.fast.scaled_dot_product_attention`
- ✅ Relative position embeddings implemented correctly
- ✅ Window attention/unpartition working
- ✅ Code follows MLX conventions
- ✅ Proper type hints and docstrings

---

### 3.2 Unit Test for SAM Encoder

**Location:** `mlx_port/tests/unit/test_sam_encoder.py`

```python
# mlx_port/tests/unit/test_sam_encoder.py
"""Unit tests for SAM vision encoder"""
import pytest
import torch
import mlx.core as mx
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "DeepSeek-OCR-master/DeepSeek-OCR-vllm"))

from deepencoder.sam_vary_sdpa import build_sam_vit_b as build_sam_pytorch
from mlx_port.deepencoder.sam_vary_mlx import build_sam_vit_b as build_sam_mlx
from mlx_port.tests.test_utils import PyTorchMLXComparator


class TestSAMEncoder:
    """Tests for SAM vision encoder"""

    @pytest.fixture
    def models(self):
        """Create PyTorch and MLX SAM models"""
        # Build models
        sam_pytorch = build_sam_pytorch()
        sam_pytorch.eval()

        sam_mlx = build_sam_mlx()

        # Load PyTorch weights into MLX
        PyTorchMLXComparator.load_pytorch_weights_to_mlx(
            sam_mlx,
            sam_pytorch.state_dict(),
            verbose=True
        )

        return sam_pytorch, sam_mlx

    def test_sam_encoder_output_shape(self, models):
        """Test SAM encoder output shape"""
        sam_pytorch, sam_mlx = models

        # Create test input
        test_input_torch = torch.randn(1, 3, 1024, 1024)
        test_input_mlx = mx.array(test_input_torch.numpy())

        # Forward pass
        with torch.no_grad():
            output_torch = sam_pytorch(test_input_torch)

        output_mlx = sam_mlx(test_input_mlx)

        # Check shapes match
        assert output_mlx.shape == list(output_torch.shape), \
            f"Shape mismatch: MLX {output_mlx.shape} vs PyTorch {output_torch.shape}"

        print(f"✓ SAM encoder output shape: {output_mlx.shape}")

    def test_sam_encoder_output_values(self, models):
        """Test SAM encoder output values match PyTorch"""
        sam_pytorch, sam_mlx = models

        # Create test input
        test_input_torch = torch.randn(2, 3, 1024, 1024)
        test_input_mlx = mx.array(test_input_torch.numpy())

        # Forward pass
        with torch.no_grad():
            output_torch = sam_pytorch(test_input_torch)

        output_mlx = sam_mlx(test_input_mlx)

        # Compare outputs
        PyTorchMLXComparator.assert_close(
            output_torch, output_mlx,
            rtol=1e-4, atol=1e-5,
            name="SAM Encoder Output",
            verbose=True
        )

    @pytest.mark.slow
    def test_sam_encoder_different_sizes(self, models):
        """Test SAM encoder with different input sizes"""
        sam_pytorch, sam_mlx = models

        sizes = [(1024, 1024), (512, 512), (640, 640)]

        for size in sizes:
            test_input_torch = torch.randn(1, 3, *size)
            test_input_mlx = mx.array(test_input_torch.numpy())

            with torch.no_grad():
                output_torch = sam_pytorch(test_input_torch)

            output_mlx = sam_mlx(test_input_mlx)

            PyTorchMLXComparator.assert_close(
                output_torch, output_mlx,
                rtol=1e-4, atol=1e-5,
                name=f"SAM Encoder ({size})",
                verbose=False
            )

        print(f"✓ SAM encoder works with different input sizes")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Run tests:**
```bash
cd mlx_port
pytest tests/unit/test_sam_encoder.py -v
```

**Acceptance Criteria:**
- ✅ SAM encoder outputs match PyTorch (rtol=1e-4, atol=1e-5)
- ✅ Works with different input sizes
- ✅ Tests pass consistently
- ✅ No shape mismatches

---

## Continuation: Phases 4-8

Due to length constraints, I'll provide a high-level overview of the remaining phases. Each follows the same pattern:

**Phase 4: CLIP Encoder** (8-10 hours)
- Similar to SAM encoder
- Different attention structure (no relative position by default)
- Uses quick_gelu activation
- Test against PyTorch implementation

**Phase 5: MLP Projector** (4-6 hours)
- Port MlpProjector class
- Integrate unfold_mlx
- Test projector outputs

**Phase 6: Main Model** (10-12 hours)
- Port DeepseekOCRForCausalLM
- Vision processing pipeline
- Multi-modal embedding merging
- Integration tests

**Phase 7: Inference Engine** (8-10 hours)
- Replace vLLM with MLX-LM
- Implement streaming generation
- Test complete pipeline

**Phase 8: Validation** (10-12 hours)
- End-to-end validation
- Performance benchmarking
- Documentation

---

## Success Metrics & Validation

**Numerical Accuracy:**
- [ ] SAM encoder: <1e-4 relative difference
- [ ] CLIP encoder: <1e-4 relative difference
- [ ] MLP projector: <1e-5 relative difference
- [ ] End-to-end: <1% OCR accuracy difference

**Performance:**
- [ ] Inference latency: <2x PyTorch on A100
- [ ] Memory usage: <1.5x PyTorch
- [ ] Throughput: >0.5 images/second

**Quality:**
- [ ] All unit tests pass (>20 tests)
- [ ] All integration tests pass (>5 tests)
- [ ] Code coverage >85%
- [ ] Documentation complete

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| SDPA performance | Benchmark early, optimize if needed |
| Weight loading | Create robust conversion utilities |
| Numerical precision | Use bfloat16, increase tolerances if needed |
| MLX API changes | Pin MLX>=0.28.0, <0.30.0 |

---

## Timeline Summary

| Phase | Duration | Cumulative |
|-------|----------|------------|
| 1. Setup | 0.5 days | 0.5 days |
| 2. Utilities | 0.5 days | 1 day |
| 3. SAM Encoder | 1.5 days | 2.5 days |
| 4. CLIP Encoder | 1 day | 3.5 days |
| 5. MLP Projector | 0.5 days | 4 days |
| 6. Main Model | 1.5 days | 5.5 days |
| 7. Inference | 1 day | 6.5 days |
| 8. Validation | 1.5 days | 8 days |
| **Total** | **8-10 days** | **58-74 hours** |

Ready to begin Phase 1! 🚀
