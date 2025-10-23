"""
Complete inference pipeline for DeepSeek-OCR MLX.

This module provides a high-level API for:
- Loading models and tokenizers
- Preprocessing images and text
- Running inference
- Generating text (with or without streaming)

Example usage:
    >>> from PIL import Image
    >>> pipeline = DeepSeekOCRPipeline.from_pretrained("deepseek-ocr")
    >>> image = Image.open("document.png")
    >>> result = pipeline(
    ...     images=[image],
    ...     prompt="Transcribe this document: <image>",
    ...     max_tokens=100
    ... )
    >>> print(result['text'])
"""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import json

import mlx.core as mx
import mlx.nn as nn
from PIL import Image

from ..models.deepseek_ocr_causal_lm_mlx import (
    DeepseekOCRForCausalLM,
    DeepseekOCRConfig,
    build_deepseek_ocr_model,
)
from ..preprocessing.image_processor_mlx import DeepseekOCRProcessor
from .generation_mlx import SamplingConfig, generate, stream_generate


def load_model_and_tokenizer(
    model_path: Union[str, Path],
    tokenizer_path: Optional[Union[str, Path]] = None,
    language_model: Optional[nn.Module] = None,
) -> tuple[DeepseekOCRForCausalLM, Any]:
    """
    Load DeepSeek-OCR model and tokenizer from disk.

    Args:
        model_path: Path to model weights directory
        tokenizer_path: Path to tokenizer (if None, uses model_path)
        language_model: Optional pre-trained language model

    Returns:
        (model, tokenizer)

    Example:
        >>> model, tokenizer = load_model_and_tokenizer("./deepseek-ocr-weights")
    """
    model_path = Path(model_path)
    tokenizer_path = Path(tokenizer_path) if tokenizer_path else model_path

    # Load config
    config_file = model_path / "config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        config = DeepseekOCRConfig(**config_dict)
    else:
        # Use default config
        config = DeepseekOCRConfig()

    # Build model
    model = build_deepseek_ocr_model(config, language_model=language_model)

    # Load weights (simplified - in production, use proper weight loading)
    weights_file = model_path / "weights.npz"
    if weights_file.exists():
        model.load_weights(str(weights_file))

    # Load tokenizer (simplified - in production, use proper tokenizer)
    # For now, we'll create a simple placeholder
    tokenizer = None  # Replace with actual tokenizer loading

    return model, tokenizer


class DeepSeekOCRPipeline:
    """
    High-level pipeline for DeepSeek-OCR inference.

    This class provides a simple interface for:
    1. Loading models and preprocessors
    2. Processing images and text
    3. Running inference
    4. Generating text

    Args:
        model: DeepSeek-OCR model
        processor: Image and text processor
        tokenizer: Tokenizer for text decoding
        device: Device to run inference on (currently unused, MLX auto-manages)

    Example:
        >>> # Create pipeline
        >>> pipeline = DeepSeekOCRPipeline(model, processor, tokenizer)
        >>>
        >>> # Run inference
        >>> result = pipeline(
        ...     images=[Image.open("doc.png")],
        ...     prompt="Transcribe: <image>",
        ...     max_tokens=100
        ... )
        >>> print(result['text'])
    """

    def __init__(
        self,
        model: DeepseekOCRForCausalLM,
        processor: DeepseekOCRProcessor,
        tokenizer: Any = None,
        device: str = "gpu",
    ):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        tokenizer_path: Optional[Union[str, Path]] = None,
        language_model: Optional[nn.Module] = None,
        **processor_kwargs
    ) -> "DeepSeekOCRPipeline":
        """
        Load pipeline from pretrained weights.

        Args:
            model_path: Path to model weights
            tokenizer_path: Path to tokenizer (optional)
            language_model: Pre-trained language model (optional)
            **processor_kwargs: Additional processor configuration

        Returns:
            DeepSeekOCRPipeline instance

        Example:
            >>> pipeline = DeepSeekOCRPipeline.from_pretrained("./weights")
        """
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(
            model_path,
            tokenizer_path,
            language_model
        )

        # Create processor
        processor = DeepseekOCRProcessor(tokenizer, **processor_kwargs)

        return cls(model, processor, tokenizer)

    def preprocess(
        self,
        images: List[Image.Image],
        prompt: str,
        cropping: bool = True,
    ) -> Dict[str, mx.array]:
        """
        Preprocess images and text.

        Args:
            images: List of PIL Images
            prompt: Text prompt with <image> placeholders
            cropping: Whether to use multi-scale cropping

        Returns:
            Dictionary with preprocessed inputs
        """
        processed = self.processor(
            prompt=prompt,
            images=images,
            cropping=cropping,
        )

        return {
            'input_ids': processed.input_ids,
            'pixel_values': processed.pixel_values,
            'images_crop': processed.images_crop,
            'images_spatial_crop': processed.images_spatial_crop,
        }

    def forward(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        images_crop: mx.array,
        images_spatial_crop: mx.array,
    ) -> mx.array:
        """
        Run forward pass through model.

        Args:
            input_ids: Token IDs
            pixel_values: Global views
            images_crop: Local crops
            images_spatial_crop: Crop grids

        Returns:
            Logits or embeddings
        """
        return self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            images_crop=images_crop,
            images_spatial_crop=images_spatial_crop,
        )

    def generate(
        self,
        images: List[Image.Image],
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
        cropping: bool = True,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate text from images and prompt.

        Args:
            images: List of PIL Images
            prompt: Text prompt with <image> placeholders
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            cropping: Whether to use multi-scale cropping
            stream: Whether to use streaming generation

        Returns:
            Dictionary with generated text and metadata

        Example:
            >>> result = pipeline.generate(
            ...     images=[Image.open("doc.png")],
            ...     prompt="Transcribe: <image>",
            ...     max_tokens=100
            ... )
            >>> print(result['text'])
        """
        # Preprocess inputs
        inputs = self.preprocess(images, prompt, cropping)

        # Process through vision models and get embeddings
        vision_embeddings = self.model.process_vision_input(
            pixel_values=inputs['pixel_values'],
            images_crop=inputs['images_crop'],
            images_spatial_crop=inputs['images_spatial_crop'],
        )

        # Get merged input embeddings
        inputs_embeds = self.model.get_input_embeddings(
            inputs['input_ids'],
            vision_embeddings,
        )

        # Configure generation
        config = SamplingConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            eos_token_id=self.processor.eos_token_id if hasattr(self.processor, 'eos_token_id') else None,
        )

        # Generate
        if stream:
            # Return generator for streaming
            return stream_generate(self.model, inputs_embeds, config, self.tokenizer)
        else:
            # Generate all at once
            generated_ids, generated_text = generate(
                self.model,
                inputs_embeds,
                config,
                self.tokenizer
            )

            return {
                'text': generated_text,
                'token_ids': generated_ids,
                'num_tokens': len(generated_ids),
            }

    def __call__(
        self,
        images: List[Image.Image],
        prompt: str,
        max_tokens: int = 512,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run inference (shorthand for generate).

        Args:
            images: List of PIL Images
            prompt: Text prompt with <image> placeholders
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with generated text and metadata
        """
        return self.generate(
            images=images,
            prompt=prompt,
            max_tokens=max_tokens,
            **kwargs
        )


def run_inference(
    model_path: str,
    images: List[Union[str, Path, Image.Image]],
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 0.9,
    cropping: bool = True,
    stream: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for running inference with minimal setup.

    Args:
        model_path: Path to model weights
        images: List of image paths or PIL Images
        prompt: Text prompt with <image> placeholders
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        cropping: Whether to use multi-scale cropping
        stream: Whether to use streaming generation
        **kwargs: Additional parameters

    Returns:
        Dictionary with generated text

    Example:
        >>> result = run_inference(
        ...     model_path="./weights",
        ...     images=["doc1.png", "doc2.png"],
        ...     prompt="Transcribe: <image> and <image>",
        ...     max_tokens=200
        ... )
        >>> print(result['text'])
    """
    # Load pipeline
    pipeline = DeepSeekOCRPipeline.from_pretrained(model_path)

    # Load images if paths provided
    loaded_images = []
    for img in images:
        if isinstance(img, (str, Path)):
            loaded_images.append(Image.open(img))
        else:
            loaded_images.append(img)

    # Run generation
    return pipeline.generate(
        images=loaded_images,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        cropping=cropping,
        stream=stream,
        **kwargs
    )
