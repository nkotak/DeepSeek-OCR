"""
Model implementations for DeepSeek-OCR MLX.

This module provides the complete multi-modal causal language model
integrating vision encoders with text generation.
"""

from .deepseek_ocr_causal_lm_mlx import (
    DeepseekOCRForCausalLM,
    build_deepseek_ocr_model,
)

__all__ = [
    'DeepseekOCRForCausalLM',
    'build_deepseek_ocr_model',
]
