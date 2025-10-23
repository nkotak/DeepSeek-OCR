#!/usr/bin/env python3
"""Validation script for Phase 5: MLP Projector Migration

This script validates the complete implementation of the MLP projector in MLX.
It checks all acceptance criteria and runs comprehensive tests.

Usage:
    python validate_phase5.py              # Run all validation checks
    python validate_phase5.py --quick      # Run quick validation only
    python validate_phase5.py --verbose    # Run with detailed output
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
        from deepencoder import projector_mlx
        from deepencoder.projector_mlx import (
            MlpProjector,
            build_linear_projector,
            build_downsample_projector,
        )
        from deepencoder.utils_mlx import unfold_mlx
        return True, "✓ All imports successful"
    except Exception as e:
        return False, f"✗ Import failed: {e}"


def check_linear_projector() -> Tuple[bool, str]:
    """Check linear projector creation and forward pass"""
    try:
        from deepencoder.projector_mlx import build_linear_projector
        import mlx.core as mx

        proj = build_linear_projector(input_dim=2048, n_embed=1280)

        # Test forward pass
        x = mx.random.normal([2, 256, 2048])
        output = proj(x)

        assert tuple(output.shape) == (2, 256, 1280), \
            f"Shape mismatch: {output.shape} vs (2, 256, 1280)"

        return True, "✓ Linear projector works: [2, 256, 2048] -> [2, 256, 1280]"
    except Exception as e:
        return False, f"✗ Linear projector check failed: {e}"


def check_mlp_projector() -> Tuple[bool, str]:
    """Check MLP projector with GELU activation"""
    try:
        from deepencoder.projector_mlx import MlpProjector
        import mlx.core as mx

        config = {
            "projector_type": "mlp_gelu",
            "input_dim": 1024,
            "n_embed": 512,
            "depth": 2,
        }

        proj = MlpProjector(config)

        # Test forward pass
        x = mx.random.normal([2, 64, 1024])
        output = proj(x)

        assert tuple(output.shape) == (2, 64, 512), \
            f"Shape mismatch: {output.shape}"

        return True, "✓ MLP GELU projector works: depth=2, GELU activation"
    except Exception as e:
        return False, f"✗ MLP projector check failed: {e}"


def check_unfold_operation() -> Tuple[bool, str]:
    """Check unfold operation for downsampling"""
    try:
        from deepencoder.utils_mlx import unfold_mlx
        import mlx.core as mx

        # Test 2x2 unfold
        x = mx.random.normal([2, 64, 16, 16])
        output = unfold_mlx(x, kernel_size=2, stride=2, padding=0)

        # Expected: [2, 64*2*2, 8*8] = [2, 256, 64]
        expected_shape = (2, 64 * 2 * 2, 8 * 8)
        assert tuple(output.shape) == expected_shape, \
            f"Unfold shape mismatch: {output.shape} vs {expected_shape}"

        return True, f"✓ Unfold operation: [2, 64, 16, 16] -> {output.shape}"
    except Exception as e:
        return False, f"✗ Unfold check failed: {e}"


def check_downsample_projector() -> Tuple[bool, str]:
    """Check downsampling projector"""
    try:
        from deepencoder.projector_mlx import build_downsample_projector
        import mlx.core as mx

        proj = build_downsample_projector(
            input_dim=512,
            n_embed=1024,
            downsample_ratio=2,
            depth=2,
            mlp_ratio=1,
            use_norm=False
        )

        # Test with 16x16 input (256 tokens)
        x = mx.random.normal([2, 256, 512])
        output = proj(x)

        # After 2x downsampling: 256 -> 64 tokens
        expected_shape = (2, 64, 1024)
        assert tuple(output.shape) == expected_shape, \
            f"Shape mismatch: {output.shape} vs {expected_shape}"

        return True, "✓ Downsample projector: 16x16 -> 8x8 (4-to-1 reduction)"
    except Exception as e:
        return False, f"✗ Downsample projector check failed: {e}"


def check_downsample_with_padding() -> Tuple[bool, str]:
    """Check downsampling with automatic padding"""
    try:
        from deepencoder.projector_mlx import MlpProjector
        import mlx.core as mx

        config = {
            "projector_type": "downsample_mlp_gelu",
            "input_dim": 256,
            "n_embed": 512,
            "downsample_ratio": 2,
            "depth": 1,
            "mlp_ratio": 1,
        }

        proj = MlpProjector(config)

        # Test with 15x15 input (not divisible by 2)
        # Should pad to 16x16
        x = mx.random.normal([1, 225, 256])  # 15*15 = 225
        output = proj(x)

        # After padding to 16x16 and downsampling: 64 tokens
        expected_shape = (1, 64, 512)
        assert tuple(output.shape) == expected_shape, \
            f"Shape mismatch: {output.shape} vs {expected_shape}"

        return True, "✓ Downsample with auto-padding: 15x15 -> 16x16 -> 8x8"
    except Exception as e:
        return False, f"✗ Padding check failed: {e}"


def check_normlayer_downsample() -> Tuple[bool, str]:
    """Check downsample projector with LayerNorm"""
    try:
        from deepencoder.projector_mlx import build_downsample_projector
        import mlx.core as mx

        proj = build_downsample_projector(
            input_dim=384,
            n_embed=768,
            downsample_ratio=2,
            depth=2,
            mlp_ratio=2,
            use_norm=True  # Enable LayerNorm
        )

        # Test forward pass
        x = mx.random.normal([2, 256, 384])
        output = proj(x)

        assert tuple(output.shape) == (2, 64, 768)

        return True, "✓ NormLayer downsample projector with LayerNorm"
    except Exception as e:
        return False, f"✗ NormLayer downsample check failed: {e}"


def check_identity_projector() -> Tuple[bool, str]:
    """Check identity projector (pass-through)"""
    try:
        from deepencoder.projector_mlx import MlpProjector
        import mlx.core as mx

        config = {
            "projector_type": "identity",
            "input_dim": 1024,
            "n_embed": 1024,
        }

        proj = MlpProjector(config)

        # Test forward pass
        x = mx.random.normal([2, 256, 1024])
        output = proj(x)

        # Should be identical (pass-through)
        assert mx.allclose(output, x, rtol=1e-7, atol=1e-8), \
            "Identity projector should not modify input"

        return True, "✓ Identity projector (pass-through, no transformation)"
    except Exception as e:
        return False, f"✗ Identity projector check failed: {e}"


def check_sam_clip_integration() -> Tuple[bool, str]:
    """Check typical SAM+CLIP to projector pipeline"""
    try:
        from deepencoder.projector_mlx import build_linear_projector
        import mlx.core as mx

        batch_size = 2

        # Simulate SAM features: [B, 1024, H, W]
        sam_features = mx.random.normal([batch_size, 1024, 16, 16])

        # Flatten SAM: [B, 1024, 16, 16] -> [B, 256, 1024]
        sam_flat = sam_features.reshape([batch_size, 1024, -1]).transpose([0, 2, 1])

        # Simulate CLIP features: [B, 257, 1024] (with CLS token)
        clip_features = mx.random.normal([batch_size, 257, 1024])

        # Remove CLS token
        clip_no_cls = clip_features[:, 1:, :]

        # Concatenate: SAM + CLIP = 2048 dim
        combined = mx.concatenate([clip_no_cls, sam_flat], axis=-1)

        # Project to language model dimension
        proj = build_linear_projector(input_dim=2048, n_embed=1280)
        output = proj(combined)

        assert tuple(output.shape) == (batch_size, 256, 1280), \
            f"Integration shape mismatch: {output.shape}"

        return True, "✓ SAM+CLIP integration: CLIP[B,256,1024] + SAM[B,256,1024] -> [B,256,1280]"
    except Exception as e:
        return False, f"✗ SAM+CLIP integration check failed: {e}"


def check_all_projector_types() -> Tuple[bool, str]:
    """Check that all projector types can be created"""
    try:
        from deepencoder.projector_mlx import MlpProjector
        import mlx.core as mx

        projector_types = [
            "identity",
            "linear",
            "mlp_gelu",
            "downsample_mlp_gelu",
            "normlayer_downsample_mlp_gelu",
        ]

        for ptype in projector_types:
            config = {
                "projector_type": ptype,
                "input_dim": 512,
                "n_embed": 1024,
                "depth": 2,
                "mlp_ratio": 1,
                "downsample_ratio": 2,
            }

            proj = MlpProjector(config)

            # Quick forward pass test
            if ptype in ["downsample_mlp_gelu", "normlayer_downsample_mlp_gelu"]:
                x = mx.random.normal([1, 64, 512])  # 8x8 spatial
            else:
                x = mx.random.normal([1, 64, 512])

            output = proj(x)
            assert output.shape[0] == 1  # Batch dimension preserved

        return True, f"✓ All projector types work: {', '.join(projector_types)}"
    except Exception as e:
        return False, f"✗ Projector types check failed: {e}"


def check_code_quality() -> Tuple[bool, str]:
    """Check code quality (docstrings, type hints)"""
    try:
        proj_file = Path(__file__).parent.parent / "deepencoder" / "projector_mlx.py"
        content = proj_file.read_text()

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

        # Check for unfold usage
        if "unfold_mlx" not in content:
            issues.append("Missing unfold_mlx usage")

        if issues:
            return False, f"✗ Code quality issues: {', '.join(issues)}"

        return True, "✓ Code quality checks passed (docstrings, type hints, imports)"
    except Exception as e:
        return False, f"✗ Code quality check failed: {e}"


def run_validation(quick: bool = False, verbose: bool = False) -> Dict[str, Tuple[bool, str]]:
    """Run all validation checks"""
    checks = [
        ("Imports", check_imports),
        ("Linear Projector", check_linear_projector),
        ("MLP Projector", check_mlp_projector),
        ("Unfold Operation", check_unfold_operation),
        ("Identity Projector", check_identity_projector),
        ("All Projector Types", check_all_projector_types),
        ("Code Quality", check_code_quality),
    ]

    if not quick:
        checks.extend([
            ("Downsample Projector", check_downsample_projector),
            ("Downsample with Padding", check_downsample_with_padding),
            ("NormLayer Downsample", check_normlayer_downsample),
            ("SAM+CLIP Integration", check_sam_clip_integration),
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
    print("PHASE 5 VALIDATION SUMMARY")
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
        ("All projector types implemented", results.get("All Projector Types", (False, ""))[0]),
        ("Linear projector works correctly", results.get("Linear Projector", (False, ""))[0]),
        ("MLP with GELU activation", results.get("MLP Projector", (False, ""))[0]),
        ("Unfold operation for downsampling", results.get("Unfold Operation", (False, ""))[0]),
        ("Downsampling with automatic padding", results.get("Downsample with Padding", (False, ""))[0]),
        ("Integration with SAM+CLIP encoders", results.get("SAM+CLIP Integration", (False, ""))[0]),
        ("Code follows MLX conventions", results.get("Code Quality", (False, ""))[0]),
    ]

    for criterion, status in criteria:
        symbol = "✅" if status else "❌"
        print(f"{symbol} {criterion}")

    print("\n" + "=" * 80)

    if passed == total:
        print("🎉 ALL VALIDATION CHECKS PASSED! Phase 5 is complete.")
    else:
        print(f"⚠️  {total - passed} checks failed. Please review and fix.")
    print("=" * 80 + "\n")

    return passed == total


def main():
    """Main validation entry point"""
    parser = argparse.ArgumentParser(description="Validate Phase 5: MLP Projector")
    parser.add_argument("--quick", action="store_true", help="Run quick validation only")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("=" * 80)
    print("PHASE 5: MLP PROJECTOR MIGRATION - VALIDATION")
    print("=" * 80)
    print("\nValidating implementation against acceptance criteria...")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")

    results = run_validation(quick=args.quick, verbose=args.verbose)
    success = print_summary(results)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
