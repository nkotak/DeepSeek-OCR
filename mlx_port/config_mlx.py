"""
Configuration for DeepSeek-OCR MLX Port

This module contains all configuration parameters for the MLX port.
"""

import os
from pathlib import Path
from typing import Optional


# ============================================================================
# Model Configuration
# ============================================================================

# Model identifier from HuggingFace
MODEL_PATH = os.environ.get('DEEPSEEK_OCR_MODEL', 'deepseek-ai/DeepSeek-OCR')

# Local cache directory for models
CACHE_DIR = Path.home() / '.cache' / 'deepseek-ocr-mlx'


# ============================================================================
# Image Processing Configuration
# ============================================================================

# Base image size (for global view)
BASE_SIZE = int(os.environ.get('BASE_SIZE', '1024'))

# Crop image size (for local views)
IMAGE_SIZE = int(os.environ.get('IMAGE_SIZE', '640'))

# Enable dynamic cropping
CROP_MODE = os.environ.get('CROP_MODE', 'true').lower() == 'true'

# Minimum number of crops
MIN_CROPS = int(os.environ.get('MIN_CROPS', '2'))

# Maximum number of crops (reduce if GPU memory limited)
MAX_CROPS = int(os.environ.get('MAX_CROPS', '6'))

# Image normalization parameters
IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)

# Patch size for vision encoders
PATCH_SIZE = 16

# Downsample ratio for vision features
DOWNSAMPLE_RATIO = 4


# ============================================================================
# Inference Configuration
# ============================================================================

# Maximum generation length
MAX_TOKENS = int(os.environ.get('MAX_TOKENS', '8192'))

# Sampling temperature (0.0 = greedy decoding)
TEMPERATURE = float(os.environ.get('TEMPERATURE', '0.0'))

# Top-p nucleus sampling
TOP_P = float(os.environ.get('TOP_P', '1.0'))

# Top-k sampling
TOP_K = int(os.environ.get('TOP_K', '0'))

# Repetition penalty
REPETITION_PENALTY = float(os.environ.get('REPETITION_PENALTY', '1.0'))

# Batch size for inference (currently only supports 1)
BATCH_SIZE = 1

# Number of workers for data loading
NUM_WORKERS = int(os.environ.get('NUM_WORKERS', '4'))


# ============================================================================
# Prompt Templates
# ============================================================================

# Default prompt template
DEFAULT_PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'

# Prompt templates for different tasks
PROMPT_TEMPLATES = {
    'document': '<image>\n<|grounding|>Convert the document to markdown.',
    'ocr': '<image>\n<|grounding|>OCR this image.',
    'free_ocr': '<image>\nFree OCR.',
    'figure': '<image>\nParse the figure.',
    'describe': '<image>\nDescribe this image in detail.',
    'locate': '<image>\nLocate <|ref|>{text}<|/ref|> in the image.',
}

# Get prompt from environment or use default
PROMPT = os.environ.get('PROMPT', DEFAULT_PROMPT)


# ============================================================================
# MLX-Specific Configuration
# ============================================================================

# Default dtype for MLX arrays
# Options: 'float32', 'float16', 'bfloat16'
MLX_DTYPE = os.environ.get('MLX_DTYPE', 'bfloat16')

# Map string dtype to MLX dtype
def get_mlx_dtype():
    """Get MLX dtype from configuration"""
    import mlx.core as mx
    dtype_map = {
        'float32': mx.float32,
        'float16': mx.float16,
        'bfloat16': mx.bfloat16,
    }
    return dtype_map.get(MLX_DTYPE, mx.bfloat16)


# MLX memory cache limit in GB
MLX_CACHE_LIMIT_GB = int(os.environ.get('MLX_CACHE_LIMIT_GB', '8'))

# Set MLX cache limit
def set_mlx_cache_limit():
    """Set MLX cache limit"""
    import mlx.core as mx
    cache_limit_bytes = MLX_CACHE_LIMIT_GB * 1024 * 1024 * 1024
    mx.metal.set_cache_limit(cache_limit_bytes)


# Enable MLX metal memory pool
ENABLE_METAL_POOL = os.environ.get('ENABLE_METAL_POOL', 'true').lower() == 'true'


# ============================================================================
# Input/Output Configuration
# ============================================================================

# Input path (set at runtime)
INPUT_PATH = os.environ.get('INPUT_PATH', '')

# Output path (set at runtime)
OUTPUT_PATH = os.environ.get('OUTPUT_PATH', './output')

# Create output directory if it doesn't exist
def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    output_path = Path(OUTPUT_PATH)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'images').mkdir(exist_ok=True)
    return output_path


# ============================================================================
# Logging and Debugging Configuration
# ============================================================================

# Print number of vision tokens
PRINT_NUM_VIS_TOKENS = os.environ.get('PRINT_NUM_VIS_TOKENS', 'false').lower() == 'true'

# Enable n-gram repetition skipping
SKIP_REPEAT = os.environ.get('SKIP_REPEAT', 'true').lower() == 'true'

# N-gram size for repetition detection
NGRAM_SIZE = int(os.environ.get('NGRAM_SIZE', '30'))

# Window size for repetition detection
WINDOW_SIZE = int(os.environ.get('WINDOW_SIZE', '90'))

# Verbose logging
VERBOSE = os.environ.get('VERBOSE', 'false').lower() == 'true'

# Log level
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')


# ============================================================================
# Testing Configuration
# ============================================================================

# Relative tolerance for numerical comparisons
TEST_TOLERANCE_RTOL = float(os.environ.get('TEST_TOLERANCE_RTOL', '1e-4'))

# Absolute tolerance for numerical comparisons
TEST_TOLERANCE_ATOL = float(os.environ.get('TEST_TOLERANCE_ATOL', '1e-5'))

# Enable validation against PyTorch
ENABLE_PYTORCH_VALIDATION = os.environ.get('ENABLE_PYTORCH_VALIDATION', 'false').lower() == 'true'


# ============================================================================
# Performance Configuration
# ============================================================================

# Enable performance profiling
ENABLE_PROFILING = os.environ.get('ENABLE_PROFILING', 'false').lower() == 'true'

# Enable timing logs
ENABLE_TIMING = os.environ.get('ENABLE_TIMING', 'false').lower() == 'true'


# ============================================================================
# Tokenizer (Lazy Loading)
# ============================================================================

_tokenizer = None

def get_tokenizer(model_path: Optional[str] = None):
    """
    Get tokenizer instance (lazy loading)

    Args:
        model_path: Path to model (defaults to MODEL_PATH)

    Returns:
        Tokenizer instance
    """
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        path = model_path or MODEL_PATH
        _tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
            cache_dir=str(CACHE_DIR)
        )
    return _tokenizer


# Tokenizer instance (property for backward compatibility)
class TokenizerProperty:
    """Property-like accessor for tokenizer"""
    def __get__(self, obj, objtype=None):
        return get_tokenizer()

TOKENIZER = TokenizerProperty()


# ============================================================================
# Vision Encoder Configuration
# ============================================================================

# SAM encoder configuration
SAM_CONFIG = {
    'img_size': 1024,
    'patch_size': 16,
    'embed_dim': 768,
    'depth': 12,
    'num_heads': 12,
    'mlp_ratio': 4.0,
    'out_chans': 256,
    'window_size': 14,
    'global_attn_indexes': [2, 5, 8, 11],
}

# CLIP encoder configuration
CLIP_CONFIG = {
    'num_layers': 24,
    'hidden_size': 1024,
    'num_attention_heads': 16,
    'ffn_hidden_size': 4096,
    'image_size': 224,
    'patch_size': 14,
    'max_position_embeddings': 256,
    'use_flash_attn': False,  # Use MLX SDPA instead
}

# Projector configuration
PROJECTOR_CONFIG = {
    'projector_type': 'linear',
    'input_dim': 2048,  # SAM (1024) + CLIP (1024)
    'n_embed': 1280,    # Output embedding dimension
}


# ============================================================================
# Utility Functions
# ============================================================================

def print_config():
    """Print current configuration"""
    print("=" * 60)
    print("DeepSeek-OCR MLX Configuration")
    print("=" * 60)
    print(f"Model Path: {MODEL_PATH}")
    print(f"Cache Dir: {CACHE_DIR}")
    print(f"\nImage Processing:")
    print(f"  Base Size: {BASE_SIZE}")
    print(f"  Image Size: {IMAGE_SIZE}")
    print(f"  Crop Mode: {CROP_MODE}")
    print(f"  Min Crops: {MIN_CROPS}")
    print(f"  Max Crops: {MAX_CROPS}")
    print(f"\nInference:")
    print(f"  Max Tokens: {MAX_TOKENS}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"\nMLX:")
    print(f"  Dtype: {MLX_DTYPE}")
    print(f"  Cache Limit: {MLX_CACHE_LIMIT_GB} GB")
    print(f"\nPaths:")
    print(f"  Input: {INPUT_PATH or '(not set)'}")
    print(f"  Output: {OUTPUT_PATH}")
    print("=" * 60)


def validate_config():
    """Validate configuration parameters"""
    errors = []

    if BASE_SIZE <= 0:
        errors.append("BASE_SIZE must be positive")
    if IMAGE_SIZE <= 0:
        errors.append("IMAGE_SIZE must be positive")
    if MIN_CROPS < 1:
        errors.append("MIN_CROPS must be >= 1")
    if MAX_CROPS < MIN_CROPS:
        errors.append("MAX_CROPS must be >= MIN_CROPS")
    if MAX_TOKENS <= 0:
        errors.append("MAX_TOKENS must be positive")
    if not 0.0 <= TEMPERATURE <= 2.0:
        errors.append("TEMPERATURE must be in [0.0, 2.0]")
    if MLX_DTYPE not in ['float32', 'float16', 'bfloat16']:
        errors.append(f"Invalid MLX_DTYPE: {MLX_DTYPE}")

    if errors:
        raise ValueError("Configuration validation failed:\n  - " + "\n  - ".join(errors))

    return True


# Validate configuration on import
try:
    validate_config()
except ValueError as e:
    import warnings
    warnings.warn(str(e), RuntimeWarning)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Model
    'MODEL_PATH',
    'CACHE_DIR',
    # Image processing
    'BASE_SIZE',
    'IMAGE_SIZE',
    'CROP_MODE',
    'MIN_CROPS',
    'MAX_CROPS',
    'IMAGE_MEAN',
    'IMAGE_STD',
    'PATCH_SIZE',
    'DOWNSAMPLE_RATIO',
    # Inference
    'MAX_TOKENS',
    'TEMPERATURE',
    'TOP_P',
    'TOP_K',
    'REPETITION_PENALTY',
    'BATCH_SIZE',
    'NUM_WORKERS',
    # Prompts
    'PROMPT',
    'PROMPT_TEMPLATES',
    'DEFAULT_PROMPT',
    # MLX
    'MLX_DTYPE',
    'MLX_CACHE_LIMIT_GB',
    'ENABLE_METAL_POOL',
    'get_mlx_dtype',
    'set_mlx_cache_limit',
    # I/O
    'INPUT_PATH',
    'OUTPUT_PATH',
    'ensure_output_dir',
    # Logging
    'PRINT_NUM_VIS_TOKENS',
    'SKIP_REPEAT',
    'NGRAM_SIZE',
    'WINDOW_SIZE',
    'VERBOSE',
    'LOG_LEVEL',
    # Testing
    'TEST_TOLERANCE_RTOL',
    'TEST_TOLERANCE_ATOL',
    'ENABLE_PYTORCH_VALIDATION',
    # Performance
    'ENABLE_PROFILING',
    'ENABLE_TIMING',
    # Tokenizer
    'TOKENIZER',
    'get_tokenizer',
    # Encoder configs
    'SAM_CONFIG',
    'CLIP_CONFIG',
    'PROJECTOR_CONFIG',
    # Utilities
    'print_config',
    'validate_config',
]
