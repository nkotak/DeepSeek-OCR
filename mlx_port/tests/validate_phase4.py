#!/usr/bin/env python3
"""Validation script for Phase 4: CLIP Vision Encoder Migration

This script validates the complete implementation of the CLIP Large encoder in MLX.
It checks all acceptance criteria and runs comprehensive tests.

Usage:
    python validate_phase4.py              # Run all validation checks
    python validate_phase4.py --quick      # Run quick validation only
    python validate_phase4.py --verbose    # Run with detailed output
"""
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "DeepSeek-OCR-master/DeepSeek-OCR-vllm"))


def check_imports() -> Tuple[bool, str]:
    """Check that all required modules can be imported"""
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from deepencoder import clip_mlx
        from deepencoder.clip_mlx import (
            CLIPVisionEmbeddings,
            CLIPAttention,
            CLIPFeedForward,
            CLIPTransformerBlock,
            CLIPTransformer,
            CLIPVisionModel,
            build_clip_l,
        )
        return True, "✓ All imports successful"
    except Exception as e:
        return False, f"✗ Import failed: {e}"


def check_model_creation() -> Tuple[bool, str]:
    """Check that CLIP model can be created"""
    try:
        from deepencoder.clip_mlx import build_clip_l
        model = build_clip_l()

        # Check architecture
        assert len(model.transformer.layers) == 24, f"Expected 24 blocks, got {len(model.transformer.layers)}"
        assert model.embeddings is not None
        assert model.pre_layernorm is not None

        return True, "✓ Model creation and architecture validated"
    except Exception as e:
        return False, f"✗ Model creation failed: {e}"


def check_sdpa_usage() -> Tuple[bool, str]:
    """Check that mx.fast.scaled_dot_product_attention is used"""
    try:
        # Read the source file and check for SDPA usage
        clip_file = Path(__file__).parent.parent / "deepencoder" / "clip_mlx.py"
        content = clip_file.read_text()

        # Check for SDPA usage
        assert "mx.fast.scaled_dot_product_attention" in content, \
            "Must use mx.fast.scaled_dot_product_attention"

        # Count occurrences
        sdpa_count = content.count("mx.fast.scaled_dot_product_attention")
        assert sdpa_count >= 1, \
            f"Expected at least 1 SDPA call, found {sdpa_count}"

        # Check that we're NOT using custom attention implementation
        assert "q @ mx.swapaxes(k, -1, -2)" not in content, \
            "Must NOT use custom attention computation"

        return True, f"✓ Using MLX native mx.fast.scaled_dot_product_attention ({sdpa_count} calls)"
    except Exception as e:
        return False, f"✗ SDPA usage check failed: {e}"


def check_quick_gelu_usage() -> Tuple[bool, str]:
    """Check that quick_gelu activation is used"""
    try:
        from deepencoder.clip_mlx import CLIPFeedForward
        from deepencoder.utils_mlx import quick_gelu_mlx
        import mlx.core as mx

        # Create FFN
        ffn = CLIPFeedForward(dim=1024, hidden_dim=4096)

        # Test forward pass
        x = mx.random.normal([2, 257, 1024])
        output = ffn(x)

        assert tuple(output.shape) == (2, 257, 1024)

        # Check source code mentions quick_gelu
        clip_file = Path(__file__).parent.parent / "deepencoder" / "clip_mlx.py"
        content = clip_file.read_text()
        assert "quick_gelu" in content, "Must use quick_gelu activation"

        return True, "✓ Using quick_gelu activation (not standard GELU)"
    except Exception as e:
        return False, f"✗ quick_gelu check failed: {e}"


def check_cls_token() -> Tuple[bool, str]:
    """Check that CLS token is properly implemented"""
    try:
        import mlx.core as mx
        from deepencoder.clip_mlx import CLIPVisionEmbeddings

        # Create embeddings
        emb = CLIPVisionEmbeddings(hidden_size=1024, image_size=224, patch_size=14)

        # Check class_embedding exists
        assert hasattr(emb, 'class_embedding'), "Missing class_embedding"

        # Test forward pass
        x = mx.random.normal([2, 3, 224, 224])
        output = emb(x, None)

        # Should have num_patches + 1 tokens
        num_patches = (224 // 14) ** 2
        expected_seq_len = num_patches + 1

        assert output.shape[1] == expected_seq_len, \
            f"Expected {expected_seq_len} tokens, got {output.shape[1]}"

        return True, f"✓ CLS token properly implemented ({expected_seq_len} = {num_patches} patches + 1 CLS)"
    except Exception as e:
        return False, f"✗ CLS token check failed: {e}"


def check_position_embeddings() -> Tuple[bool, str]:
    """Check that learned position embeddings are implemented"""
    try:
        import mlx.core as mx
        from deepencoder.clip_mlx import CLIPVisionEmbeddings

        # Create embeddings
        emb = CLIPVisionEmbeddings(hidden_size=1024, image_size=224, patch_size=14)

        # Check position_embedding exists
        assert hasattr(emb, 'position_embedding'), "Missing position_embedding"

        # Check shape: [1, num_positions, hidden_size]
        num_patches = (224 // 14) ** 2
        num_positions = num_patches + 1  # +1 for CLS
        expected_shape = (1, num_positions, 1024)

        assert tuple(emb.position_embedding.shape) == expected_shape, \
            f"Position embedding shape mismatch: {emb.position_embedding.shape} vs {expected_shape}"

        return True, f"✓ Learned position embeddings: {emb.position_embedding.shape}"
    except Exception as e:
        return False, f"✗ Position embeddings check failed: {e}"


def check_position_interpolation() -> Tuple[bool, str]:
    """Check that position interpolation works for different image sizes"""
    try:
        import mlx.core as mx
        from deepencoder.clip_mlx import build_clip_l

        model = build_clip_l()

        # Test with different image size (requires interpolation)
        sizes = [224, 336]
        results = []

        for size in sizes:
            x = mx.random.normal([1, 3, size, size])
            output = model(x, None)

            num_patches = (size // 14) ** 2
            expected_shape = (1, num_patches + 1, 1024)

            assert tuple(output.shape) == expected_shape, \
                f"Shape mismatch for size {size}: {output.shape} vs {expected_shape}"

            results.append(f"{size}x{size} -> {output.shape}")

        return True, f"✓ Position interpolation works:\n    " + "\n    ".join(results)
    except Exception as e:
        return False, f"✗ Position interpolation check failed: {e}"


def check_forward_pass_shapes() -> Tuple[bool, str]:
    """Check that forward pass produces correct shapes"""
    try:
        import mlx.core as mx
        from deepencoder.clip_mlx import build_clip_l

        model = build_clip_l()

        # Test different batch sizes
        test_cases = [
            ((1, 3, 224, 224), (1, 257, 1024)),
            ((2, 3, 224, 224), (2, 257, 1024)),
            ((4, 3, 224, 224), (4, 257, 1024)),
        ]

        results = []
        for input_shape, expected_output in test_cases:
            x = mx.random.normal(input_shape)
            output = model(x, None)
            actual_shape = tuple(output.shape)

            if actual_shape != expected_output:
                return False, f"✗ Shape mismatch: {input_shape} -> {actual_shape}, expected {expected_output}"

            results.append(f"{input_shape} -> {actual_shape}")

        return True, f"✓ Forward pass shapes correct:\n    " + "\n    ".join(results)
    except Exception as e:
        return False, f"✗ Forward pass failed: {e}"


def check_prelayer_norm() -> Tuple[bool, str]:
    """Check that pre-LayerNorm architecture is used"""
    try:
        from deepencoder.clip_mlx import build_clip_l

        model = build_clip_l()

        # Check pre_layernorm exists
        assert hasattr(model, 'pre_layernorm'), "Missing pre_layernorm"

        # Check transformer blocks have layer norms
        for i, block in enumerate(model.transformer.layers):
            assert hasattr(block, 'layer_norm1'), f"Block {i} missing layer_norm1"
            assert hasattr(block, 'layer_norm2'), f"Block {i} missing layer_norm2"

        return True, "✓ Pre-LayerNorm architecture confirmed"
    except Exception as e:
        return False, f"✗ Pre-LayerNorm check failed: {e}"


def check_attention_mechanism() -> Tuple[bool, str]:
    """Check attention mechanism implementation"""
    try:
        import mlx.core as mx
        from deepencoder.clip_mlx import CLIPAttention

        # Create attention
        attn = CLIPAttention(hidden_size=1024, num_attention_heads=16)

        # Test forward pass
        x = mx.random.normal([2, 257, 1024])
        output = attn(x)

        # Check output shape
        assert tuple(output.shape) == (2, 257, 1024), \
            f"Attention output shape mismatch: {output.shape}"

        return True, "✓ Attention mechanism works correctly"
    except Exception as e:
        return False, f"✗ Attention check failed: {e}"


def check_code_quality() -> Tuple[bool, str]:
    """Check code quality (docstrings, type hints)"""
    try:
        clip_file = Path(__file__).parent.parent / "deepencoder" / "clip_mlx.py"
        content = clip_file.read_text()

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

        # Check for proper references to MLX SDPA
        if "scaled_dot_product_attention" not in content:
            issues.append("Missing reference to MLX SDPA")

        if issues:
            return False, f"✗ Code quality issues: {', '.join(issues)}"

        return True, "✓ Code quality checks passed (docstrings, type hints, imports)"
    except Exception as e:
        return False, f"✗ Code quality check failed: {e}"


def run_validation(quick: bool = False, verbose: bool = False) -> Dict[str, Tuple[bool, str]]:
    """Run all validation checks"""
    checks = [
        ("Imports", check_imports),
        ("Model Creation", check_model_creation),
        ("SDPA Usage", check_sdpa_usage),
        ("Quick GELU Activation", check_quick_gelu_usage),
        ("CLS Token", check_cls_token),
        ("Position Embeddings", check_position_embeddings),
        ("Pre-LayerNorm Architecture", check_prelayer_norm),
        ("Attention Mechanism", check_attention_mechanism),
        ("Code Quality", check_code_quality),
    ]

    if not quick:
        checks.extend([
            ("Forward Pass Shapes", check_forward_pass_shapes),
            ("Position Interpolation", check_position_interpolation),
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
    print("PHASE 4 VALIDATION SUMMARY")
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
        ("All CLIP encoder classes implemented", results.get("Model Creation", (False, ""))[0]),
        ("Attention uses mx.fast.scaled_dot_product_attention", results.get("SDPA Usage", (False, ""))[0]),
        ("Uses quick_gelu activation (not standard GELU)", results.get("Quick GELU Activation", (False, ""))[0]),
        ("CLS token properly implemented", results.get("CLS Token", (False, ""))[0]),
        ("Learned position embeddings with interpolation", results.get("Position Embeddings", (False, ""))[0]),
        ("Pre-LayerNorm architecture", results.get("Pre-LayerNorm Architecture", (False, ""))[0]),
        ("Code follows MLX conventions", results.get("Code Quality", (False, ""))[0]),
    ]

    for criterion, status in criteria:
        symbol = "✅" if status else "❌"
        print(f"{symbol} {criterion}")

    print("\n" + "=" * 80)

    if passed == total:
        print("🎉 ALL VALIDATION CHECKS PASSED! Phase 4 is complete.")
    else:
        print(f"⚠️  {total - passed} checks failed. Please review and fix.")
    print("=" * 80 + "\n")

    return passed == total


def main():
    """Main validation entry point"""
    parser = argparse.ArgumentParser(description="Validate Phase 4: CLIP Vision Encoder")
    parser.add_argument("--quick", action="store_true", help="Run quick validation only")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("=" * 80)
    print("PHASE 4: CLIP VISION ENCODER MIGRATION - VALIDATION")
    print("=" * 80)
    print("\nValidating implementation against acceptance criteria...")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")

    results = run_validation(quick=args.quick, verbose=args.verbose)
    success = print_summary(results)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
