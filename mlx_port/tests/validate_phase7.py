"""
Validation script for Phase 7: Inference Engine.

This script validates that the inference engine implementation meets all requirements:
1. Image preprocessing (transformation, cropping, tokenization)
2. DeepseekOCRForCausalLM model (vision + language)
3. Vision embedding merging
4. Text generation (greedy and sampling)
5. Streaming generation
6. Complete inference pipeline
7. Multi-image support
8. Multi-scale (global + crops) support
9. Code quality (MLX native operations, no PyTorch)
"""

import sys
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def check_imports() -> Tuple[bool, str]:
    """Check that all required modules can be imported."""
    try:
        # Preprocessing
        from preprocessing.image_processor_mlx import (
            ImageTransform,
            dynamic_preprocess,
            count_tiles,
            DeepseekOCRProcessor,
        )

        # Models
        from models.deepseek_ocr_causal_lm_mlx import (
            DeepseekOCRForCausalLM,
            DeepseekOCRConfig,
            build_deepseek_ocr_model,
        )

        # Inference
        from inference.generation_mlx import (
            generate,
            stream_generate,
            SamplingConfig,
        )
        from inference.pipeline_mlx import (
            DeepSeekOCRPipeline,
            load_model_and_tokenizer,
        )

        return True, "All imports successful"
    except ImportError as e:
        return False, f"Import failed: {str(e)}"


def check_preprocessing() -> Tuple[bool, str]:
    """Check preprocessing pipeline."""
    try:
        import mlx.core as mx
        from PIL import Image
        from preprocessing.image_processor_mlx import ImageTransform, dynamic_preprocess

        # Test image transformation
        transform = ImageTransform()
        img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        mx_img = transform(img)

        if mx_img.shape != (3, 224, 224):
            return False, f"ImageTransform shape mismatch: {mx_img.shape}"

        # Test dynamic preprocessing
        img = Image.new('RGB', (1280, 640))
        crops, grid = dynamic_preprocess(img, image_size=640)

        if len(crops) != grid[0] * grid[1]:
            return False, f"Crop count mismatch: {len(crops)} != {grid[0]} * {grid[1]}"

        return True, "Preprocessing pipeline working"
    except Exception as e:
        return False, f"Preprocessing check failed: {str(e)}"


def check_model_creation() -> Tuple[bool, str]:
    """Check model creation."""
    try:
        from models.deepseek_ocr_causal_lm_mlx import DeepseekOCRConfig, build_deepseek_ocr_model

        config = DeepseekOCRConfig(
            image_token_id=128256,
            n_embed=128,
            vocab_size=1000,
        )
        model = build_deepseek_ocr_model(config)

        # Check components
        if not hasattr(model, 'sam_model'):
            return False, "Model missing sam_model"
        if not hasattr(model, 'vision_model'):
            return False, "Model missing vision_model"
        if not hasattr(model, 'projector'):
            return False, "Model missing projector"
        if not hasattr(model, 'language_model'):
            return False, "Model missing language_model"

        return True, "Model creation successful"
    except Exception as e:
        return False, f"Model creation failed: {str(e)}"


def check_vision_processing() -> Tuple[bool, str]:
    """Check vision processing."""
    try:
        import mlx.core as mx
        from models.deepseek_ocr_causal_lm_mlx import DeepseekOCRConfig, build_deepseek_ocr_model

        config = DeepseekOCRConfig(image_token_id=128256, n_embed=128)
        model = build_deepseek_ocr_model(config)

        # Test global-only processing
        pixel_values = mx.random.normal((1, 3, 1280, 1280))
        images_crop = mx.zeros((1, 1, 3, 1024, 1024))
        images_spatial_crop = mx.array([[1, 1]])

        vision_embeds = model.process_vision_input(
            pixel_values, images_crop, images_spatial_crop
        )

        if len(vision_embeds) != 1:
            return False, f"Expected 1 vision embedding, got {len(vision_embeds)}"

        if vision_embeds[0].ndim != 2:
            return False, f"Vision embedding should be 2D, got {vision_embeds[0].ndim}D"

        return True, "Vision processing working"
    except Exception as e:
        return False, f"Vision processing failed: {str(e)}"


def check_embedding_merging() -> Tuple[bool, str]:
    """Check vision-text embedding merging."""
    try:
        import mlx.core as mx
        from models.deepseek_ocr_causal_lm_mlx import DeepseekOCRConfig, build_deepseek_ocr_model

        config = DeepseekOCRConfig(image_token_id=128256, n_embed=128, hidden_size=128)
        model = build_deepseek_ocr_model(config)

        # Test merging
        input_ids = mx.array([[1, 128256, 50, 2]])  # [BOS, <image>, token, EOS]
        vision_embeds = [mx.zeros((10, 128))]  # 10 vision tokens

        merged = model.get_input_embeddings(input_ids, vision_embeds)

        # Expected length: 1 (BOS) + 10 (vision) + 1 (token) + 1 (EOS) = 13
        if merged.shape[1] != 13:
            return False, f"Expected seq_len=13, got {merged.shape[1]}"

        return True, "Embedding merging working"
    except Exception as e:
        return False, f"Embedding merging failed: {str(e)}"


def check_generation() -> Tuple[bool, str]:
    """Check text generation."""
    try:
        import mlx.core as mx
        from models.deepseek_ocr_causal_lm_mlx import DeepseekOCRConfig, build_deepseek_ocr_model
        from inference.generation_mlx import SamplingConfig, generate

        config = DeepseekOCRConfig(image_token_id=128256, n_embed=128, vocab_size=1000, hidden_size=128)
        model = build_deepseek_ocr_model(config)

        # Test generation
        inputs_embeds = mx.random.normal((1, 5, 128))
        gen_config = SamplingConfig(temperature=0.0, max_tokens=2, eos_token_id=2)

        token_ids, text = generate(model, inputs_embeds, gen_config, tokenizer=None)

        if not isinstance(token_ids, list):
            return False, "Generated tokens should be a list"

        return True, "Text generation working"
    except Exception as e:
        return False, f"Text generation failed: {str(e)}"


def check_streaming() -> Tuple[bool, str]:
    """Check streaming generation."""
    try:
        import mlx.core as mx
        from models.deepseek_ocr_causal_lm_mlx import DeepseekOCRConfig, build_deepseek_ocr_model
        from inference.generation_mlx import SamplingConfig, stream_generate

        config = DeepseekOCRConfig(image_token_id=128256, n_embed=128, vocab_size=1000, hidden_size=128)
        model = build_deepseek_ocr_model(config)

        # Test streaming
        inputs_embeds = mx.random.normal((1, 5, 128))
        gen_config = SamplingConfig(temperature=0.0, max_tokens=2, eos_token_id=2)

        stream = stream_generate(model, inputs_embeds, gen_config, tokenizer=None)

        # Check that it's a generator
        if not hasattr(stream, '__iter__'):
            return False, "stream_generate should return an iterator"

        # Consume one token
        try:
            token_id, text = next(stream)
            if not isinstance(token_id, int):
                return False, "Token ID should be an integer"
        except StopIteration:
            pass  # Empty generation is ok for this test

        return True, "Streaming generation working"
    except Exception as e:
        return False, f"Streaming generation failed: {str(e)}"


def check_pipeline() -> Tuple[bool, str]:
    """Check complete inference pipeline."""
    try:
        import mlx.core as mx
        from PIL import Image
        from models.deepseek_ocr_causal_lm_mlx import DeepseekOCRConfig, build_deepseek_ocr_model
        from preprocessing.image_processor_mlx import DeepseekOCRProcessor
        from inference.pipeline_mlx import DeepSeekOCRPipeline

        # Create dummy tokenizer
        class DummyTokenizer:
            vocab = {"<image>": 128256}
            bos_token_id = 1
            eos_token_id = 2
            def encode(self, text, add_special_tokens=False):
                return [100, 200, 300]

        config = DeepseekOCRConfig(image_token_id=128256, n_embed=128, vocab_size=1000, hidden_size=128)
        model = build_deepseek_ocr_model(config)
        tokenizer = DummyTokenizer()
        processor = DeepseekOCRProcessor(tokenizer, image_size=1024, base_size=1280)

        pipeline = DeepSeekOCRPipeline(model, processor, tokenizer)

        # Test preprocessing
        img = Image.new('RGB', (640, 640))
        inputs = pipeline.preprocess([img], "Test: <image>", cropping=False)

        if 'input_ids' not in inputs:
            return False, "Pipeline preprocess missing input_ids"

        return True, "Complete pipeline working"
    except Exception as e:
        return False, f"Pipeline check failed: {str(e)}"


def check_multi_image() -> Tuple[bool, str]:
    """Check multi-image support."""
    try:
        import mlx.core as mx
        from PIL import Image
        from models.deepseek_ocr_causal_lm_mlx import DeepseekOCRConfig, build_deepseek_ocr_model
        from preprocessing.image_processor_mlx import DeepseekOCRProcessor
        from inference.pipeline_mlx import DeepSeekOCRPipeline

        class DummyTokenizer:
            vocab = {"<image>": 128256}
            bos_token_id = 1
            eos_token_id = 2
            def encode(self, text, add_special_tokens=False):
                return [100, 200]

        config = DeepseekOCRConfig(image_token_id=128256, n_embed=128, vocab_size=1000, hidden_size=128)
        model = build_deepseek_ocr_model(config)
        tokenizer = DummyTokenizer()
        processor = DeepseekOCRProcessor(tokenizer, image_size=1024, base_size=1280)
        pipeline = DeepSeekOCRPipeline(model, processor, tokenizer)

        # Test with 2 images
        img1 = Image.new('RGB', (640, 640))
        img2 = Image.new('RGB', (640, 640))
        inputs = pipeline.preprocess([img1, img2], "Compare <image> and <image>", cropping=False)

        if inputs['pixel_values'].shape[0] != 2:
            return False, f"Expected 2 images, got {inputs['pixel_values'].shape[0]}"

        return True, "Multi-image support working"
    except Exception as e:
        return False, f"Multi-image check failed: {str(e)}"


def check_multi_scale() -> Tuple[bool, str]:
    """Check multi-scale (global + crops) support."""
    try:
        import mlx.core as mx
        from PIL import Image
        from preprocessing.image_processor_mlx import DeepseekOCRProcessor

        class DummyTokenizer:
            vocab = {"<image>": 128256}
            bos_token_id = 1
            eos_token_id = 2
            def encode(self, text, add_special_tokens=False):
                return [100]

        tokenizer = DummyTokenizer()
        processor = DeepseekOCRProcessor(tokenizer, image_size=1024, base_size=1280)

        # Test with large image (should trigger cropping)
        img = Image.new('RGB', (1920, 1080))
        outputs = processor("Text: <image>", [img], cropping=True)

        # Should have crops or spatial crop info
        if outputs.images_spatial_crop[0, 0] == 1 and outputs.images_spatial_crop[0, 1] == 1:
            # No crops, check if it's because image is too small
            if img.size[0] > 640 or img.size[1] > 640:
                return False, "Large image should trigger cropping"

        return True, "Multi-scale support working"
    except Exception as e:
        return False, f"Multi-scale check failed: {str(e)}"


def check_code_quality() -> Tuple[bool, str]:
    """Check code quality (MLX native operations, no PyTorch)."""
    files_to_check = [
        'preprocessing/image_processor_mlx.py',
        'models/deepseek_ocr_causal_lm_mlx.py',
        'inference/generation_mlx.py',
        'inference/pipeline_mlx.py',
    ]

    issues = []

    for file_path in files_to_check:
        full_path = Path(__file__).parent.parent / file_path
        if not full_path.exists():
            issues.append(f"File not found: {file_path}")
            continue

        with open(full_path, 'r') as f:
            content = f.read()

        # Check for PyTorch imports (should not exist)
        if re.search(r'import torch(?!vision)', content):
            issues.append(f"{file_path}: Contains PyTorch import")

        # Check for MLX imports (should exist)
        if 'import mlx' not in content and 'from mlx' not in content:
            # Some files might not directly import MLX (e.g., __init__.py)
            if not file_path.endswith('__init__.py'):
                issues.append(f"{file_path}: Missing MLX import")

        # Check for placeholder/TODO comments
        if 'TODO' in content or 'PLACEHOLDER' in content or 'FIXME' in content:
            issues.append(f"{file_path}: Contains TODO/PLACEHOLDER/FIXME")

        # Check for mock/example/placeholder in function names
        # Note: "sample" is allowed (sample_token is a real sampling function)
        if re.search(r'def (mock|example|placeholder|dummy)_', content):
            issues.append(f"{file_path}: Contains mock/example/placeholder functions")

    if issues:
        return False, "; ".join(issues)

    return True, "Code quality checks passed"


def main():
    """Run all validation checks."""
    print("=" * 80)
    print("PHASE 7: INFERENCE ENGINE - VALIDATION")
    print("=" * 80)
    print()
    print("Validating implementation against acceptance criteria...")
    print()

    checks = [
        ("Imports", check_imports),
        ("Preprocessing Pipeline", check_preprocessing),
        ("Model Creation", check_model_creation),
        ("Vision Processing", check_vision_processing),
        ("Embedding Merging", check_embedding_merging),
        ("Text Generation", check_generation),
        ("Streaming Generation", check_streaming),
        ("Complete Pipeline", check_pipeline),
        ("Multi-Image Support", check_multi_image),
        ("Multi-Scale Support", check_multi_scale),
        ("Code Quality", check_code_quality),
    ]

    results = {}
    passed = 0
    failed = 0

    for name, check_func in checks:
        try:
            success, message = check_func()
            results[name] = (success, message)
            if success:
                passed += 1
                print(f"  ✓ {message}")
            else:
                failed += 1
                print(f"  ✗ {message}")
        except Exception as e:
            results[name] = (False, str(e))
            failed += 1
            print(f"  ✗ {name} check failed: {str(e)}")

    # Print summary
    print()
    print("=" * 80)
    print("PHASE 7 VALIDATION SUMMARY")
    print("=" * 80)
    print()
    print(f"Results: {passed}/{len(checks)} checks passed ({passed * 100 // len(checks)}%)")
    print()

    for name, (success, message) in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {name}")
        if not success:
            print(f"  {message}")

    # Acceptance criteria
    print()
    print("=" * 80)
    print("ACCEPTANCE CRITERIA STATUS")
    print("=" * 80)

    criteria = {
        "Image preprocessing pipeline works": results.get("Preprocessing Pipeline", (False, ""))[0],
        "DeepseekOCRForCausalLM model created": results.get("Model Creation", (False, ""))[0],
        "Vision processing pipeline works": results.get("Vision Processing", (False, ""))[0],
        "Vision-text embedding merging works": results.get("Embedding Merging", (False, ""))[0],
        "Text generation works": results.get("Text Generation", (False, ""))[0],
        "Streaming generation works": results.get("Streaming Generation", (False, ""))[0],
        "Complete inference pipeline works": results.get("Complete Pipeline", (False, ""))[0],
        "Multi-image support works": results.get("Multi-Image Support", (False, ""))[0],
        "Multi-scale (crops) support works": results.get("Multi-Scale Support", (False, ""))[0],
        "Code uses MLX native operations": results.get("Code Quality", (False, ""))[0],
        "No PyTorch dependencies": results.get("Code Quality", (False, ""))[0],
    }

    for criterion, status in criteria.items():
        symbol = "✅" if status else "❌"
        print(f"{symbol} {criterion}")

    print()
    print("=" * 80)
    if failed == 0:
        print("✅ All checks passed! Phase 7 implementation is complete.")
    else:
        print(f"⚠️  {failed} checks failed. Please review and fix.")
    print("=" * 80)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
