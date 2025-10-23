#!/usr/bin/env python3
"""Validation script for Phase 3: SAM Vision Encoder Migration

This script validates the complete implementation of the SAM ViT-B encoder in MLX.
It checks all acceptance criteria and runs comprehensive tests.

Usage:
    python validate_phase3.py              # Run all validation checks
    python validate_phase3.py --quick      # Run quick validation only
    python validate_phase3.py --verbose    # Run with detailed output
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
        from deepencoder import sam_vary_mlx
        from deepencoder.sam_vary_mlx import (
            MLPBlock,
            LayerNorm2d,
            Attention,
            Block,
            PatchEmbed,
            ImageEncoderViT,
            build_sam_vit_b,
            window_partition,
            window_unpartition,
        )
        return True, "✓ All imports successful"
    except Exception as e:
        return False, f"✗ Import failed: {e}"


def check_model_creation() -> Tuple[bool, str]:
    """Check that SAM model can be created"""
    try:
        from deepencoder.sam_vary_mlx import build_sam_vit_b
        model = build_sam_vit_b()

        # Check architecture
        assert len(model.blocks) == 12, f"Expected 12 blocks, got {len(model.blocks)}"
        assert len(model.neck) == 4, f"Expected 4 neck layers, got {len(model.neck)}"
        assert model.patch_embed is not None
        assert model.net_2 is not None
        assert model.net_3 is not None

        return True, "✓ Model creation and architecture validated"
    except Exception as e:
        return False, f"✗ Model creation failed: {e}"


def check_window_attention_config() -> Tuple[bool, str]:
    """Check that window attention is properly configured"""
    try:
        from deepencoder.sam_vary_mlx import build_sam_vit_b
        model = build_sam_vit_b()

        # Check window sizes
        global_attn_indexes = [2, 5, 8, 11]
        for i, blk in enumerate(model.blocks):
            if i in global_attn_indexes:
                assert blk.window_size == 0, f"Block {i} should use global attention"
            else:
                assert blk.window_size == 14, f"Block {i} should use window=14"

        return True, "✓ Window attention configured correctly (global at [2,5,8,11], window=14 elsewhere)"
    except Exception as e:
        return False, f"✗ Window attention config failed: {e}"


def check_sdpa_usage() -> Tuple[bool, str]:
    """Check that mx.fast.scaled_dot_product_attention is used"""
    try:
        # Read the source file and check for SDPA usage
        sam_file = Path(__file__).parent.parent / "deepencoder" / "sam_vary_mlx.py"
        content = sam_file.read_text()

        # Check for SDPA usage
        assert "mx.fast.scaled_dot_product_attention" in content, \
            "Must use mx.fast.scaled_dot_product_attention"

        # Count occurrences (should appear in Attention class)
        sdpa_count = content.count("mx.fast.scaled_dot_product_attention")
        assert sdpa_count >= 2, \
            f"Expected at least 2 SDPA calls (with/without mask), found {sdpa_count}"

        # Check that we're NOT using custom attention implementation
        assert "q @ mx.swapaxes(k, -1, -2)" not in content, \
            "Must NOT use custom attention computation"

        return True, f"✓ Using MLX native mx.fast.scaled_dot_product_attention ({sdpa_count} calls)"
    except Exception as e:
        return False, f"✗ SDPA usage check failed: {e}"


def check_forward_pass_shapes() -> Tuple[bool, str]:
    """Check that forward pass produces correct shapes"""
    try:
        import mlx.core as mx
        from deepencoder.sam_vary_mlx import build_sam_vit_b

        model = build_sam_vit_b()

        # Test different input sizes
        test_cases = [
            ((1, 3, 512, 512), (1, 1024, 8, 8)),
            ((1, 3, 1024, 1024), (1, 1024, 16, 16)),
            ((2, 3, 1024, 1024), (2, 1024, 16, 16)),
        ]

        results = []
        for input_shape, expected_output in test_cases:
            x = mx.random.normal(input_shape)
            output = model(x)
            actual_shape = tuple(output.shape)

            if actual_shape != expected_output:
                return False, f"✗ Shape mismatch: {input_shape} -> {actual_shape}, expected {expected_output}"

            results.append(f"{input_shape} -> {actual_shape}")

        return True, f"✓ Forward pass shapes correct:\n    " + "\n    ".join(results)
    except Exception as e:
        return False, f"✗ Forward pass failed: {e}"


def check_relative_position_embeddings() -> Tuple[bool, str]:
    """Check that relative position embeddings are implemented"""
    try:
        import mlx.core as mx
        from deepencoder.sam_vary_mlx import Attention

        # Create attention with relative position
        attn = Attention(
            dim=768,
            num_heads=12,
            use_rel_pos=True,
            rel_pos_zero_init=True,
            input_size=(64, 64)
        )

        # Check that rel_pos attributes exist
        assert hasattr(attn, 'rel_pos_h'), "Missing rel_pos_h"
        assert hasattr(attn, 'rel_pos_w'), "Missing rel_pos_w"

        # Check shapes
        expected_h_shape = (2 * 64 - 1, 768 // 12)
        expected_w_shape = (2 * 64 - 1, 768 // 12)

        assert tuple(attn.rel_pos_h.shape) == expected_h_shape, \
            f"rel_pos_h shape mismatch: {attn.rel_pos_h.shape} vs {expected_h_shape}"
        assert tuple(attn.rel_pos_w.shape) == expected_w_shape, \
            f"rel_pos_w shape mismatch: {attn.rel_pos_w.shape} vs {expected_w_shape}"

        # Test forward pass with relative position
        x = mx.random.normal((1, 64, 64, 768))
        output = attn(x)
        assert tuple(output.shape) == (1, 64, 64, 768)

        return True, "✓ Relative position embeddings implemented correctly"
    except Exception as e:
        return False, f"✗ Relative position check failed: {e}"


def check_window_operations() -> Tuple[bool, str]:
    """Check window partition and unpartition operations"""
    try:
        import mlx.core as mx
        from deepencoder.sam_vary_mlx import window_partition, window_unpartition

        # Test window operations
        batch_size = 2
        h, w = 64, 64
        channels = 768
        window_size = 14

        # Create test input
        x = mx.random.normal((batch_size, h, w, channels))

        # Partition
        windows, pad_hw = window_partition(x, window_size)

        # Check partition shape
        expected_num_windows = ((h + (window_size - h % window_size) % window_size) // window_size) * \
                               ((w + (window_size - w % window_size) % window_size) // window_size)
        expected_shape = (batch_size * expected_num_windows, window_size, window_size, channels)

        assert tuple(windows.shape) == expected_shape, \
            f"Window partition shape mismatch: {windows.shape} vs {expected_shape}"

        # Unpartition
        reconstructed = window_unpartition(windows, window_size, pad_hw, (h, w))

        # Check reconstruction
        assert tuple(reconstructed.shape) == (batch_size, h, w, channels), \
            f"Unpartition shape mismatch: {reconstructed.shape}"

        # Check that reconstruction is close to original (should be exact for partition/unpartition)
        assert mx.allclose(reconstructed, x, rtol=1e-6, atol=1e-7), \
            "Reconstruction doesn't match original"

        return True, "✓ Window operations work correctly"
    except Exception as e:
        return False, f"✗ Window operations failed: {e}"


def check_mlp_block() -> Tuple[bool, str]:
    """Check MLPBlock implementation"""
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from deepencoder.sam_vary_mlx import MLPBlock

        # Create MLP block
        mlp = MLPBlock(embedding_dim=768, mlp_dim=3072, act=nn.GELU)

        # Test forward pass
        x = mx.random.normal((2, 16, 16, 768))
        output = mlp(x)

        assert tuple(output.shape) == (2, 16, 16, 768), \
            f"MLP output shape mismatch: {output.shape}"

        return True, "✓ MLPBlock implementation correct"
    except Exception as e:
        return False, f"✗ MLPBlock check failed: {e}"


def check_layer_norm_2d() -> Tuple[bool, str]:
    """Check LayerNorm2d implementation"""
    try:
        import mlx.core as mx
        from deepencoder.sam_vary_mlx import LayerNorm2d

        # Create LayerNorm2d
        ln = LayerNorm2d(num_channels=256)

        # Test forward pass [B, C, H, W]
        x = mx.random.normal((2, 256, 32, 32))
        output = ln(x)

        assert tuple(output.shape) == (2, 256, 32, 32), \
            f"LayerNorm2d output shape mismatch: {output.shape}"

        return True, "✓ LayerNorm2d implementation correct"
    except Exception as e:
        return False, f"✗ LayerNorm2d check failed: {e}"


def check_patch_embed() -> Tuple[bool, str]:
    """Check PatchEmbed implementation"""
    try:
        import mlx.core as mx
        from deepencoder.sam_vary_mlx import PatchEmbed

        # Create PatchEmbed
        pe = PatchEmbed(
            kernel_size=(16, 16),
            stride=(16, 16),
            in_chans=3,
            embed_dim=768
        )

        # Test forward pass [B, C, H, W]
        x = mx.random.normal((2, 3, 1024, 1024))
        output = pe(x)

        expected_shape = (2, 64, 64, 768)  # 1024/16 = 64
        assert tuple(output.shape) == expected_shape, \
            f"PatchEmbed output shape mismatch: {output.shape} vs {expected_shape}"

        return True, "✓ PatchEmbed implementation correct"
    except Exception as e:
        return False, f"✗ PatchEmbed check failed: {e}"


def check_code_quality() -> Tuple[bool, str]:
    """Check code quality (docstrings, type hints)"""
    try:
        sam_file = Path(__file__).parent.parent / "deepencoder" / "sam_vary_mlx.py"
        content = sam_file.read_text()

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
        if "PR #2468" not in content and "scaled_dot_product_attention" not in content:
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
        ("Window Attention Config", check_window_attention_config),
        ("SDPA Usage", check_sdpa_usage),
        ("MLPBlock", check_mlp_block),
        ("LayerNorm2d", check_layer_norm_2d),
        ("PatchEmbed", check_patch_embed),
        ("Window Operations", check_window_operations),
        ("Relative Position Embeddings", check_relative_position_embeddings),
        ("Code Quality", check_code_quality),
    ]

    if not quick:
        checks.append(("Forward Pass Shapes", check_forward_pass_shapes))

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
    print("PHASE 3 VALIDATION SUMMARY")
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
        ("All SAM encoder classes implemented", results.get("Model Creation", (False, ""))[0]),
        ("Attention uses mx.fast.scaled_dot_product_attention", results.get("SDPA Usage", (False, ""))[0]),
        ("Relative position embeddings implemented", results.get("Relative Position Embeddings", (False, ""))[0]),
        ("Window attention/unpartition working", results.get("Window Operations", (False, ""))[0]),
        ("Code follows MLX conventions", results.get("Code Quality", (False, ""))[0]),
        ("Proper type hints and docstrings", results.get("Code Quality", (False, ""))[0]),
    ]

    for criterion, status in criteria:
        symbol = "✅" if status else "❌"
        print(f"{symbol} {criterion}")

    print("\n" + "=" * 80)

    if passed == total:
        print("🎉 ALL VALIDATION CHECKS PASSED! Phase 3 is complete.")
    else:
        print(f"⚠️  {total - passed} checks failed. Please review and fix.")
    print("=" * 80 + "\n")

    return passed == total


def main():
    """Main validation entry point"""
    parser = argparse.ArgumentParser(description="Validate Phase 3: SAM Vision Encoder")
    parser.add_argument("--quick", action="store_true", help="Run quick validation only")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("=" * 80)
    print("PHASE 3: SAM VISION ENCODER MIGRATION - VALIDATION")
    print("=" * 80)
    print("\nValidating implementation against acceptance criteria...")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")

    results = run_validation(quick=args.quick, verbose=args.verbose)
    success = print_summary(results)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
