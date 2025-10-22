"""
MLX Port of DeepSeek-OCR

This package contains the MLX (Apple Silicon) port of DeepSeek-OCR,
enabling OCR inference on Apple Silicon hardware using Metal acceleration.

Modules:
    deepencoder: Vision encoders (SAM, CLIP) and projectors
    process: Image processing and preprocessing
    tests: Unit, integration, and validation tests
    benchmarks: Performance benchmarking utilities

Usage:
    from mlx_port import DeepseekOCRMLX

    model = DeepseekOCRMLX.from_pretrained('deepseek-ai/DeepSeek-OCR')
    result = model.infer(image_path, prompt="<image>\\nOCR this image.")
"""

__version__ = "0.1.0"
__author__ = "DeepSeek-OCR MLX Port"

# Version check for MLX
try:
    import mlx.core as mx
    MLX_VERSION = mx.__version__

    # Check minimum version
    major, minor = map(int, MLX_VERSION.split('.')[:2])
    if major == 0 and minor < 28:
        import warnings
        warnings.warn(
            f"MLX version {MLX_VERSION} detected. "
            "DeepSeek-OCR MLX port requires MLX >=0.28.0 for SDPA support. "
            "Some features may not work correctly.",
            RuntimeWarning
        )
except ImportError:
    import warnings
    warnings.warn(
        "MLX not found. Please install with: pip install mlx>=0.28.0",
        RuntimeWarning
    )
    MLX_VERSION = None

__all__ = [
    '__version__',
    'MLX_VERSION',
]
