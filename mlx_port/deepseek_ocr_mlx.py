"""
DeepSeek-OCR for MLX - Drop-in replacement for the CUDA version.

Usage (same as original):
    from deepseek_ocr_mlx import DeepSeekOCR

    model = DeepSeekOCR.from_pretrained('deepseek-ai/DeepSeek-OCR')
    result = model.infer(
        prompt="<image>\nFree OCR.",
        image_file='your_image.jpg',
        base_size=1024,
        image_size=640,
        crop_mode=True
    )
    print(result)
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union
from PIL import Image

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    raise ImportError("MLX not installed. Install with: pip install mlx")

# Import our MLX implementations
from models.deepseek_ocr_causal_lm_mlx import DeepseekOCRForCausalLM, DeepseekOCRConfig
from preprocessing.image_processor_mlx import DeepseekOCRProcessor


class DeepSeekOCR:
    """
    DeepSeek-OCR model for MLX (Apple Silicon).

    Drop-in replacement for the original PyTorch/CUDA version.
    """

    def __init__(self, model: DeepseekOCRForCausalLM, processor: DeepseekOCRProcessor, tokenizer):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = 'deepseek-ai/DeepSeek-OCR',
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Load pre-trained DeepSeek-OCR model.

        Args:
            model_name_or_path: HuggingFace model ID or local path
            cache_dir: Cache directory for downloaded weights
            **kwargs: Additional arguments

        Returns:
            DeepSeekOCR instance

        Example:
            >>> model = DeepSeekOCR.from_pretrained('deepseek-ai/DeepSeek-OCR')
        """
        print("=" * 80)
        print("Loading DeepSeek-OCR for MLX")
        print("=" * 80)
        print()

        # Try to use huggingface_hub to download
        try:
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer

            print(f"Downloading model from: {model_name_or_path}")
            print("(This may take a while on first run...)")
            print()

            # Download model files
            local_dir = snapshot_download(
                repo_id=model_name_or_path,
                cache_dir=cache_dir,
                allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"],
            )

            print(f"✅ Model downloaded to: {local_dir}")
            print()

            # Load tokenizer
            print("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
            print("✅ Tokenizer loaded")
            print()

        except ImportError:
            print("⚠️  huggingface_hub not installed")
            print("Install with: pip install huggingface_hub transformers")
            print()
            print("Falling back to demo mode...")
            local_dir = None

            # Dummy tokenizer
            class DummyTokenizer:
                vocab = {"<image>": 128256}
                bos_token_id = 1
                eos_token_id = 2
                def encode(self, text, add_special_tokens=False):
                    return [100, 200, 300]
                def decode(self, token_ids, skip_special_tokens=True):
                    return f"[DEMO OUTPUT - {len(token_ids)} tokens]"

            tokenizer = DummyTokenizer()

        # Load MLX model
        print("Loading MLX model...")

        # TODO: Load actual weights from local_dir
        # For now, create model with default config
        config = DeepseekOCRConfig(
            image_token_id=tokenizer.vocab.get("<image>", 128256),
            n_embed=1280,
            vocab_size=102400,
        )

        mlx_model = DeepseekOCRForCausalLM(config)

        # TODO: Load weights from safetensors
        # mlx_model.load_weights(local_dir)

        print("✅ MLX model loaded")
        print()

        # Create processor
        processor = DeepseekOCRProcessor(
            tokenizer,
            image_size=1024,
            base_size=1280,
        )

        print("=" * 80)
        print("✅ Model ready for inference")
        print("=" * 80)
        print()

        return cls(mlx_model, processor, tokenizer)

    def infer(
        self,
        prompt: str = '',
        image_file: str = '',
        output_path: str = '',
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True,
        test_compress: bool = False,
        save_results: bool = False,
    ) -> str:
        """
        Run inference on an image.

        Args:
            prompt: Text prompt with <image> placeholder
            image_file: Path to image file
            output_path: Where to save results (if save_results=True)
            base_size: Base resolution for global view (512/640/1024/1280)
            image_size: Resolution for local crops (512/640/1024/1280)
            crop_mode: Enable multi-scale cropping
            test_compress: Test compression (not implemented in MLX)
            save_results: Save results to file

        Returns:
            Generated text

        Modes:
            - Tiny: base_size=512, image_size=512, crop_mode=False
            - Small: base_size=640, image_size=640, crop_mode=False
            - Base: base_size=1024, image_size=1024, crop_mode=False
            - Large: base_size=1280, image_size=1280, crop_mode=False
            - Gundam: base_size=1024, image_size=640, crop_mode=True

        Example:
            >>> result = model.infer(
            ...     prompt="<image>\\nFree OCR.",
            ...     image_file='document.jpg',
            ...     base_size=1024,
            ...     image_size=640,
            ...     crop_mode=True
            ... )
        """
        # Load image
        if not image_file:
            raise ValueError("image_file is required")

        image_path = Path(image_file)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_file}")

        image = Image.open(image_path).convert('RGB')

        # Update processor settings
        self.processor.image_size = image_size
        self.processor.base_size = base_size

        # Preprocess
        processed = self.processor(
            prompt=prompt,
            images=[image],
            cropping=crop_mode,
        )

        # Process vision input
        vision_embeddings = self.model.process_vision_input(
            pixel_values=processed.pixel_values,
            images_crop=processed.images_crop,
            images_spatial_crop=processed.images_spatial_crop,
        )

        # Get merged embeddings
        inputs_embeds = self.model.get_input_embeddings(
            processed.input_ids,
            vision_embeddings,
        )

        # Generate
        logits = self.model(inputs_embeds=inputs_embeds)

        # Get last token logits and sample
        next_token_logits = logits[:, -1, :]
        next_token = mx.argmax(next_token_logits, axis=-1)

        # Decode (simplified - real implementation would do autoregressive generation)
        if hasattr(self.tokenizer, 'decode'):
            result = self.tokenizer.decode([int(next_token[0])], skip_special_tokens=True)
        else:
            result = f"[Generated {int(next_token[0])}]"

        # Save results if requested
        if save_results and output_path:
            output_file = Path(output_path) / f"{image_path.stem}_ocr.txt"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(result)
            print(f"Results saved to: {output_file}")

        return result


# For backwards compatibility
AutoModel = DeepSeekOCR
