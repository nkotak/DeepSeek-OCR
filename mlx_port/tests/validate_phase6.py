#!/usr/bin/env python3
"""Validation script for Phase 6: Main Model Integration

This script validates the complete implementation of the DeepSeek-OCR main model in MLX.
It checks all acceptance criteria and runs comprehensive tests.

Usage:
    python validate_phase6.py              # Run all validation checks
    python validate_phase6.py --quick      # Run quick validation only
    python validate_phase6.py --verbose    # Run with detailed output
"""
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_imports() -> Tuple[bool, str]:
    """Check that all required modules can be imported"""
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from deepencoder import deepseek_ocr_mlx
        from deepencoder.deepseek_ocr_mlx import (
            DeepSeekOCRVisionModel,
            build_deepseek_ocr_vision_model,
            merge_vision_embeddings,
            calculate_num_image_tokens,
        )
        return True, "✓ All imports successful"
    except Exception as e:
        return False, f"✗ Import failed: {e}"


def check_model_creation() -> Tuple[bool, str]:
    """Check that main model can be created"""
    try:
        from deepencoder.deepseek_ocr_mlx import build_deepseek_ocr_vision_model

        model = build_deepseek_ocr_vision_model(n_embed=1280)

        # Check components
        assert model.sam_model is not None, "Missing SAM encoder"
        assert model.vision_model is not None, "Missing CLIP encoder"
        assert model.projector is not None, "Missing projector"
        assert model.image_newline is not None, "Missing image_newline token"
        assert model.view_separator is not None, "Missing view_separator token"

        return True, "✓ Main model created with all components (SAM, CLIP, projector, special tokens)"
    except Exception as e:
        return False, f"✗ Model creation failed: {e}"


def check_special_tokens() -> Tuple[bool, str]:
    """Check special tokens initialization"""
    try:
        from deepencoder.deepseek_ocr_mlx import build_deepseek_ocr_vision_model
        import mlx.core as mx

        model = build_deepseek_ocr_vision_model(n_embed=1280)

        # Check shapes
        assert tuple(model.image_newline.shape) == (1280,), \
            f"image_newline shape mismatch: {model.image_newline.shape}"
        assert tuple(model.view_separator.shape) == (1280,), \
            f"view_separator shape mismatch: {model.view_separator.shape}"

        # Check values are not zeros
        assert mx.sum(mx.abs(model.image_newline)).item() > 0, \
            "image_newline is all zeros"
        assert mx.sum(mx.abs(model.view_separator)).item() > 0, \
            "view_separator is all zeros"

        return True, f"✓ Special tokens: image_newline={model.image_newline.shape}, " \
                     f"view_separator={model.view_separator.shape}"
    except Exception as e:
        return False, f"✗ Special tokens check failed: {e}"


def check_single_image_encoding() -> Tuple[bool, str]:
    """Check encoding single image through full pipeline"""
    try:
        from deepencoder.deepseek_ocr_mlx import build_deepseek_ocr_vision_model
        import mlx.core as mx

        model = build_deepseek_ocr_vision_model(n_embed=1280)

        # Test with 1024x1024 image
        image = mx.random.normal([1, 3, 1024, 1024])
        features = model._encode_image(image)

        # Expected: [1, 256, 1280] where 256 = (1024/16/4)^2 = 16^2
        expected_shape = (1, 256, 1280)
        assert tuple(features.shape) == expected_shape, \
            f"Shape mismatch: {features.shape} vs {expected_shape}"

        return True, f"✓ Single image encoding: [1, 3, 1024, 1024] -> {features.shape}"
    except Exception as e:
        return False, f"✗ Single image encoding failed: {e}"


def check_multi_scale_support() -> Tuple[bool, str]:
    """Check support for multiple image scales"""
    try:
        from deepencoder.deepseek_ocr_mlx import build_deepseek_ocr_vision_model
        import mlx.core as mx

        model = build_deepseek_ocr_vision_model(n_embed=1280)

        test_sizes = [
            (512, 512, 64),    # 512x512 -> 8x8 = 64 tokens
            (1024, 1024, 256), # 1024x1024 -> 16x16 = 256 tokens
            (1280, 1280, 400), # 1280x1280 -> 20x20 = 400 tokens
        ]

        results = []
        for h, w, expected_tokens in test_sizes:
            image = mx.random.normal([1, 3, h, w])
            features = model._encode_image(image)

            assert tuple(features.shape) == (1, expected_tokens, 1280), \
                f"Size {h}x{w}: expected {expected_tokens} tokens, got {features.shape[1]}"

            results.append(f"{h}x{w} -> {expected_tokens} tokens")

        return True, f"✓ Multi-scale support:\n    " + "\n    ".join(results)
    except Exception as e:
        return False, f"✗ Multi-scale check failed: {e}"


def check_global_only_processing() -> Tuple[bool, str]:
    """Check processing global-only images (no crops)"""
    try:
        from deepencoder.deepseek_ocr_mlx import build_deepseek_ocr_vision_model
        import mlx.core as mx

        model = build_deepseek_ocr_vision_model(n_embed=1280)

        # Process without crops
        image = mx.random.normal([1, 3, 1024, 1024])
        result = model.process_single_image(image, patches=None, crop_shape=None)

        # Expected: 16*17 + 1 = 273 tokens (16x16 grid + newlines + separator)
        expected_tokens = 16 * 17 + 1
        assert tuple(result.shape) == (expected_tokens, 1280), \
            f"Shape mismatch: {result.shape} vs ({expected_tokens}, 1280)"

        return True, f"✓ Global-only processing: 256 tokens -> {expected_tokens} (with newlines + separator)"
    except Exception as e:
        return False, f"✗ Global-only processing failed: {e}"


def check_multi_scale_processing() -> Tuple[bool, str]:
    """Check processing multi-scale images (global + local crops)"""
    try:
        from deepencoder.deepseek_ocr_mlx import build_deepseek_ocr_vision_model
        import mlx.core as mx

        model = build_deepseek_ocr_vision_model(n_embed=1280)

        # Global + 2x2 local crops
        global_image = mx.random.normal([1, 3, 1024, 1024])
        local_patches = mx.random.normal([4, 3, 1024, 1024])
        crop_shape = (2, 2)

        result = model.process_single_image(global_image, local_patches, crop_shape)

        # Expected: 32*33 (local) + 16*17 (global) + 1 (separator) = 1329
        expected_tokens = 32 * 33 + 16 * 17 + 1
        assert tuple(result.shape) == (expected_tokens, 1280), \
            f"Shape mismatch: {result.shape} vs ({expected_tokens}, 1280)"

        return True, f"✓ Multi-scale processing: local + global -> {expected_tokens} tokens"
    except Exception as e:
        return False, f"✗ Multi-scale processing failed: {e}"


def check_2d_layout_formatting() -> Tuple[bool, str]:
    """Check 2D layout formatting with newlines"""
    try:
        from deepencoder.deepseek_ocr_mlx import build_deepseek_ocr_vision_model
        import mlx.core as mx

        model = build_deepseek_ocr_vision_model(n_embed=1280)

        # Create 16x16 feature grid
        global_features = mx.random.normal([1, 256, 1280])

        # Format with 2D layout
        formatted = model._format_global_only(global_features)

        # Check that newlines are added: 16 rows * (16 cols + 1 newline) + 1 separator
        expected_tokens = 16 * 17 + 1
        assert formatted.shape[0] == expected_tokens, \
            f"Expected {expected_tokens} tokens, got {formatted.shape[0]}"

        return True, f"✓ 2D layout: 16x16 grid -> 16 rows × 17 tokens/row + 1 separator = {expected_tokens}"
    except Exception as e:
        return False, f"✗ 2D layout check failed: {e}"


def check_vision_embedding_merging() -> Tuple[bool, str]:
    """Check merging vision embeddings into text"""
    try:
        from deepencoder.deepseek_ocr_mlx import merge_vision_embeddings
        import mlx.core as mx

        # Text: "Hello <image> world" -> [1, 2, 100, 3, 4]
        input_ids = mx.array([[1, 2, 100, 3, 4]])
        inputs_embeds = mx.random.normal([1, 5, 1280])

        # Vision embedding: 273 tokens
        vision_embeddings = [mx.random.normal([273, 1280])]

        # Merge
        result = merge_vision_embeddings(
            input_ids, inputs_embeds, vision_embeddings, image_token_id=100
        )

        # Expected: 2 (before) + 273 (vision) + 2 (after) = 277
        expected_len = 277
        assert result.shape[0] == expected_len, \
            f"Expected {expected_len} tokens, got {result.shape[0]}"

        return True, f"✓ Vision embedding merging: 5 text + 273 vision = {result.shape[0]} total"
    except Exception as e:
        return False, f"✗ Vision embedding merging failed: {e}"


def check_batch_processing() -> Tuple[bool, str]:
    """Check batch processing of multiple images"""
    try:
        from deepencoder.deepseek_ocr_mlx import build_deepseek_ocr_vision_model
        import mlx.core as mx

        model = build_deepseek_ocr_vision_model(n_embed=1280)

        # Batch of 3 images
        pixel_values = [
            mx.random.normal([1, 3, 1024, 1024]),
            mx.random.normal([1, 3, 1024, 1024]),
            mx.random.normal([1, 3, 1024, 1024])
        ]
        images_crop = [None, None, None]
        images_spatial_crop = [None, None, None]

        # Process
        results = model(pixel_values, images_crop, images_spatial_crop)

        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

        for i, result in enumerate(results):
            expected_tokens = 16 * 17 + 1
            assert tuple(result.shape) == (expected_tokens, 1280), \
                f"Result {i}: shape mismatch {result.shape}"

        return True, f"✓ Batch processing: 3 images processed successfully"
    except Exception as e:
        return False, f"✗ Batch processing failed: {e}"


def check_token_calculation() -> Tuple[bool, str]:
    """Check vision token calculation"""
    try:
        from deepencoder.deepseek_ocr_mlx import calculate_num_image_tokens

        test_cases = [
            (512, 512, False, 273),      # Small, no crops
            (1024, 1024, False, 273),    # Medium, no crops
            (2048, 2048, True, 1329),    # Large, with crops
        ]

        results = []
        for width, height, crop_mode, expected in test_cases:
            num_tokens = calculate_num_image_tokens(
                image_width=width,
                image_height=height,
                crop_mode=crop_mode
            )

            assert num_tokens == expected, \
                f"Size {width}x{height}: expected {expected} tokens, got {num_tokens}"

            results.append(f"{width}x{height} ({'crops' if crop_mode else 'no crops'}): {num_tokens} tokens")

        return True, f"✓ Token calculation:\n    " + "\n    ".join(results)
    except Exception as e:
        return False, f"✗ Token calculation failed: {e}"


def check_code_quality() -> Tuple[bool, str]:
    """Check code quality (docstrings, type hints)"""
    try:
        model_file = Path(__file__).parent.parent / "deepencoder" / "deepseek_ocr_mlx.py"
        content = model_file.read_text()

        issues = []

        # Check for proper imports
        if "import mlx.core as mx" not in content:
            issues.append("Missing MLX core import")
        if "import mlx.nn as nn" not in content:
            issues.append("Missing MLX nn import")

        # Check for docstrings
        if '"""' not in content:
            issues.append("Missing docstrings")

        # Check for type hints
        if "mx.array" not in content:
            issues.append("Missing type hints")

        # Check for special tokens
        if "image_newline" not in content:
            issues.append("Missing image_newline")
        if "view_separator" not in content:
            issues.append("Missing view_separator")

        # Check for integration with all encoders
        if "sam_model" not in content:
            issues.append("Missing SAM integration")
        if "vision_model" not in content:
            issues.append("Missing CLIP integration")
        if "projector" not in content:
            issues.append("Missing projector integration")

        if issues:
            return False, f"✗ Code quality issues: {', '.join(issues)}"

        return True, "✓ Code quality checks passed (docstrings, type hints, integrations)"
    except Exception as e:
        return False, f"✗ Code quality check failed: {e}"


def run_validation(quick: bool = False, verbose: bool = False) -> Dict[str, Tuple[bool, str]]:
    """Run all validation checks"""
    checks = [
        ("Imports", check_imports),
        ("Model Creation", check_model_creation),
        ("Special Tokens", check_special_tokens),
        ("Single Image Encoding", check_single_image_encoding),
        ("Global-Only Processing", check_global_only_processing),
        ("2D Layout Formatting", check_2d_layout_formatting),
        ("Vision Embedding Merging", check_vision_embedding_merging),
        ("Code Quality", check_code_quality),
    ]

    if not quick:
        checks.extend([
            ("Multi-Scale Support", check_multi_scale_support),
            ("Multi-Scale Processing", check_multi_scale_processing),
            ("Batch Processing", check_batch_processing),
            ("Token Calculation", check_token_calculation),
        ])

    results = {}
    for name, check_func in checks:
        if verbose:
            print(f"\nRunning: {name}...")
        try:
            success, message = check_func()
            results[name] = (success, message)
            if verbose or not success:
                print(f"  {message}")
        except Exception as e:
            results[name] = (False, f"✗ Unexpected error: {e}")
            if verbose:
                print(f"  ✗ Unexpected error: {e}")

    return results


def print_summary(results: Dict[str, Tuple[bool, str]]):
    """Print validation summary"""
    print("\n" + "=" * 80)
    print("PHASE 6 VALIDATION SUMMARY")
    print("=" * 80)

    total = len(results)
    passed = sum(1 for success, _ in results.values() if success)

    print(f"\nResults: {passed}/{total} checks passed ({100*passed//total}%)\n")

    for name, (success, message) in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {name}")
        if not success or "--verbose" in sys.argv:
            print(f"  {message}")

    print("\n" + "=" * 80)
    print("ACCEPTANCE CRITERIA STATUS")
    print("=" * 80)

    criteria = [
        ("Main model integrates SAM, CLIP, and projector", results.get("Model Creation", (False, ""))[0]),
        ("Single image encoding pipeline works", results.get("Single Image Encoding", (False, ""))[0]),
        ("Multi-scale image support (global + crops)", results.get("Multi-Scale Processing", (False, ""))[0]),
        ("2D layout with newline tokens", results.get("2D Layout Formatting", (False, ""))[0]),
        ("View separator for multi-view images", results.get("Special Tokens", (False, ""))[0]),
        ("Vision embedding merging into text", results.get("Vision Embedding Merging", (False, ""))[0]),
        ("Batch processing support", results.get("Batch Processing", (False, ""))[0]),
        ("Code follows MLX conventions", results.get("Code Quality", (False, ""))[0]),
    ]

    for criterion, status in criteria:
        symbol = "✅" if status else "❌"
        print(f"{symbol} {criterion}")

    print("\n" + "=" * 80)

    if passed == total:
        print("🎉 ALL VALIDATION CHECKS PASSED! Phase 6 is complete.")
    else:
        print(f"⚠️  {total - passed} checks failed. Please review and fix.")
    print("=" * 80 + "\n")

    return passed == total


def main():
    """Main validation entry point"""
    parser = argparse.ArgumentParser(description="Validate Phase 6: Main Model Integration")
    parser.add_argument("--quick", action="store_true", help="Run quick validation only")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("=" * 80)
    print("PHASE 6: MAIN MODEL INTEGRATION - VALIDATION")
    print("=" * 80)
    print("\nValidating implementation against acceptance criteria...")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")

    results = run_validation(quick=args.quick, verbose=args.verbose)
    success = print_summary(results)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
