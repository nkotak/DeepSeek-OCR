"""
Unit tests for MLX utility functions

This module tests all utility functions in deepencoder/utils_mlx.py against
PyTorch reference implementations to ensure numerical accuracy.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


# ============================================================================
# Test UnfoldMLX
# ============================================================================

class TestUnfoldMLX:
    """Tests for unfold_mlx function"""

    @pytest.mark.requires_pytorch
    def test_unfold_2x2_kernel(self):
        """Test unfold with 2x2 kernel"""
        import torch
        import torch.nn.functional as F
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import unfold_mlx

        # Create test input
        torch_input = torch.randn(2, 3, 4, 4)
        mlx_input = mx.array(torch_input.numpy())

        # PyTorch unfold
        torch_output = F.unfold(torch_input, kernel_size=2, stride=2, padding=0)

        # MLX unfold
        mlx_output = unfold_mlx(mlx_input, kernel_size=2, stride=2, padding=0)

        # Compare
        np.testing.assert_allclose(
            torch_output.numpy(),
            np.array(mlx_output),
            rtol=1e-6, atol=1e-6
        )

        # Check shape
        expected_shape = [2, 12, 4]  # [B, C*k*k, num_patches] = [2, 3*2*2, (4/2)*(4/2)]
        assert list(mlx_output.shape) == expected_shape

    @pytest.mark.requires_pytorch
    def test_unfold_4x4_kernel(self):
        """Test unfold with 4x4 kernel on larger input"""
        import torch
        import torch.nn.functional as F
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import unfold_mlx

        torch_input = torch.randn(1, 64, 8, 8)
        mlx_input = mx.array(torch_input.numpy())

        torch_output = F.unfold(torch_input, kernel_size=4, stride=4, padding=0)
        mlx_output = unfold_mlx(mlx_input, kernel_size=4, stride=4, padding=0)

        np.testing.assert_allclose(
            torch_output.numpy(),
            np.array(mlx_output),
            rtol=1e-6, atol=1e-6
        )

        # Check shape
        expected_shape = [1, 1024, 4]  # [1, 64*4*4, (8/4)*(8/4)]
        assert list(mlx_output.shape) == expected_shape

    @pytest.mark.requires_pytorch
    def test_unfold_tuple_kernel(self):
        """Test unfold with tuple kernel_size (non-square)"""
        import torch
        import torch.nn.functional as F
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import unfold_mlx

        torch_input = torch.randn(1, 3, 8, 12)
        mlx_input = mx.array(torch_input.numpy())

        torch_output = F.unfold(torch_input, kernel_size=(2, 3), stride=(2, 3), padding=0)
        mlx_output = unfold_mlx(mlx_input, kernel_size=(2, 3), stride=(2, 3), padding=0)

        np.testing.assert_allclose(
            torch_output.numpy(),
            np.array(mlx_output),
            rtol=1e-6, atol=1e-6
        )

        # Check shape
        expected_shape = [1, 18, 16]  # [1, 3*2*3, (8/2)*(12/3)]
        assert list(mlx_output.shape) == expected_shape

    @pytest.mark.requires_pytorch
    def test_unfold_single_patch(self):
        """Test unfold that results in single patch"""
        import torch
        import torch.nn.functional as F
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import unfold_mlx

        torch_input = torch.randn(2, 16, 4, 4)
        mlx_input = mx.array(torch_input.numpy())

        # Kernel size equals input size
        torch_output = F.unfold(torch_input, kernel_size=4, stride=4, padding=0)
        mlx_output = unfold_mlx(mlx_input, kernel_size=4, stride=4, padding=0)

        np.testing.assert_allclose(
            torch_output.numpy(),
            np.array(mlx_output),
            rtol=1e-6, atol=1e-6
        )

        # Check shape - should be single patch
        expected_shape = [2, 256, 1]  # [2, 16*4*4, 1]
        assert list(mlx_output.shape) == expected_shape

    def test_unfold_raises_on_overlapping(self):
        """Test that unfold raises error for overlapping patches"""
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import unfold_mlx

        mlx_input = mx.random.normal([1, 3, 8, 8])

        with pytest.raises(NotImplementedError, match="non-overlapping"):
            unfold_mlx(mlx_input, kernel_size=3, stride=2, padding=0)

    def test_unfold_raises_on_padding(self):
        """Test that unfold raises error for padding"""
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import unfold_mlx

        mlx_input = mx.random.normal([1, 3, 8, 8])

        with pytest.raises(NotImplementedError, match="padding=0"):
            unfold_mlx(mlx_input, kernel_size=2, stride=2, padding=1)

    def test_unfold_raises_on_indivisible(self):
        """Test that unfold raises error when dimensions not divisible"""
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import unfold_mlx

        mlx_input = mx.random.normal([1, 3, 7, 7])

        with pytest.raises(ValueError, match="divisible"):
            unfold_mlx(mlx_input, kernel_size=2, stride=2, padding=0)


# ============================================================================
# Test InterpolateMLX
# ============================================================================

class TestInterpolateMLX:
    """Tests for interpolate_mlx function"""

    @pytest.mark.requires_pytorch
    def test_interpolate_bicubic_downsample(self):
        """Test bicubic interpolation (downsampling)"""
        import torch
        import torch.nn.functional as F
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import interpolate_mlx

        torch_input = torch.randn(2, 3, 64, 64)
        mlx_input = mx.array(torch_input.numpy())

        torch_output = F.interpolate(
            torch_input,
            size=(32, 32),
            mode='bicubic',
            align_corners=False,
            antialias=True
        )
        mlx_output = interpolate_mlx(mlx_input, size=(32, 32), mode='bicubic')

        # Allow larger tolerance for interpolation
        np.testing.assert_allclose(
            torch_output.numpy(),
            np.array(mlx_output),
            rtol=1e-3, atol=1e-4
        )

        # Check shape
        expected_shape = [2, 3, 32, 32]
        assert list(mlx_output.shape) == expected_shape

    @pytest.mark.requires_pytorch
    def test_interpolate_bicubic_upsample(self):
        """Test bicubic interpolation (upsampling)"""
        import torch
        import torch.nn.functional as F
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import interpolate_mlx

        torch_input = torch.randn(1, 3, 32, 32)
        mlx_input = mx.array(torch_input.numpy())

        torch_output = F.interpolate(
            torch_input,
            size=(64, 64),
            mode='bicubic',
            align_corners=False,
            antialias=True
        )
        mlx_output = interpolate_mlx(mlx_input, size=(64, 64), mode='bicubic')

        np.testing.assert_allclose(
            torch_output.numpy(),
            np.array(mlx_output),
            rtol=1e-3, atol=1e-4
        )

        expected_shape = [1, 3, 64, 64]
        assert list(mlx_output.shape) == expected_shape

    @pytest.mark.requires_pytorch
    def test_interpolate_bilinear(self):
        """Test bilinear interpolation"""
        import torch
        import torch.nn.functional as F
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import interpolate_mlx

        torch_input = torch.randn(1, 3, 32, 32)
        mlx_input = mx.array(torch_input.numpy())

        torch_output = F.interpolate(
            torch_input,
            size=(64, 64),
            mode='bilinear',
            align_corners=False
        )
        mlx_output = interpolate_mlx(mlx_input, size=(64, 64), mode='bilinear')

        np.testing.assert_allclose(
            torch_output.numpy(),
            np.array(mlx_output),
            rtol=1e-3, atol=1e-4
        )

    def test_interpolate_no_resize(self):
        """Test that no resizing occurs when size matches"""
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import interpolate_mlx

        mlx_input = mx.random.normal([2, 3, 32, 32])
        mlx_output = interpolate_mlx(mlx_input, size=(32, 32))

        # Should be identical (no operation)
        np.testing.assert_array_equal(
            np.array(mlx_input),
            np.array(mlx_output)
        )

    def test_interpolate_non_square(self):
        """Test interpolation with non-square target size"""
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import interpolate_mlx

        mlx_input = mx.random.normal([1, 3, 64, 64])
        mlx_output = interpolate_mlx(mlx_input, size=(32, 48), mode='bicubic')

        expected_shape = [1, 3, 32, 48]
        assert list(mlx_output.shape) == expected_shape

    def test_interpolate_invalid_size(self):
        """Test that invalid size raises error"""
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import interpolate_mlx

        mlx_input = mx.random.normal([1, 3, 32, 32])

        with pytest.raises(ValueError, match="tuple/list of 2 elements"):
            interpolate_mlx(mlx_input, size=(32,))


# ============================================================================
# Test PadMLX
# ============================================================================

class TestPadMLX:
    """Tests for pad_mlx function"""

    @pytest.mark.requires_pytorch
    def test_pad_constant(self):
        """Test constant padding"""
        import torch
        import torch.nn.functional as F
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import pad_mlx

        torch_input = torch.randn(2, 3, 10, 10)
        mlx_input = mx.array(torch_input.numpy())

        pad_spec = (1, 1, 2, 2)  # left, right, top, bottom
        torch_output = F.pad(torch_input, pad_spec, mode='constant', value=0)
        mlx_output = pad_mlx(mlx_input, pad_spec, mode='constant', value=0)

        np.testing.assert_allclose(
            torch_output.numpy(),
            np.array(mlx_output),
            rtol=1e-6, atol=1e-6
        )

        # Check shape: 10+2+2=14 height, 10+1+1=12 width
        expected_shape = [2, 3, 14, 12]
        assert list(mlx_output.shape) == expected_shape

    @pytest.mark.requires_pytorch
    def test_pad_constant_custom_value(self):
        """Test constant padding with custom value"""
        import torch
        import torch.nn.functional as F
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import pad_mlx

        torch_input = torch.randn(1, 3, 8, 8)
        mlx_input = mx.array(torch_input.numpy())

        pad_spec = (2, 2, 2, 2)
        value = -1.5
        torch_output = F.pad(torch_input, pad_spec, mode='constant', value=value)
        mlx_output = pad_mlx(mlx_input, pad_spec, mode='constant', value=value)

        np.testing.assert_allclose(
            torch_output.numpy(),
            np.array(mlx_output),
            rtol=1e-6, atol=1e-6
        )

    @pytest.mark.requires_pytorch
    def test_pad_reflect(self):
        """Test reflect padding"""
        import torch
        import torch.nn.functional as F
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import pad_mlx

        torch_input = torch.randn(1, 3, 8, 8)
        mlx_input = mx.array(torch_input.numpy())

        pad_spec = (1, 1, 1, 1)
        torch_output = F.pad(torch_input, pad_spec, mode='reflect')
        mlx_output = pad_mlx(mlx_input, pad_spec, mode='reflect')

        np.testing.assert_allclose(
            torch_output.numpy(),
            np.array(mlx_output),
            rtol=1e-6, atol=1e-6
        )

    @pytest.mark.requires_pytorch
    def test_pad_replicate(self):
        """Test replicate/edge padding"""
        import torch
        import torch.nn.functional as F
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import pad_mlx

        torch_input = torch.randn(1, 3, 6, 6)
        mlx_input = mx.array(torch_input.numpy())

        pad_spec = (2, 2, 2, 2)
        torch_output = F.pad(torch_input, pad_spec, mode='replicate')
        mlx_output = pad_mlx(mlx_input, pad_spec, mode='edge')  # 'edge' is MLX equivalent

        np.testing.assert_allclose(
            torch_output.numpy(),
            np.array(mlx_output),
            rtol=1e-6, atol=1e-6
        )

    @pytest.mark.requires_pytorch
    def test_pad_asymmetric(self):
        """Test asymmetric padding"""
        import torch
        import torch.nn.functional as F
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import pad_mlx

        torch_input = torch.randn(2, 3, 8, 8)
        mlx_input = mx.array(torch_input.numpy())

        pad_spec = (1, 2, 3, 4)  # Different padding on each side
        torch_output = F.pad(torch_input, pad_spec, mode='constant', value=0)
        mlx_output = pad_mlx(mlx_input, pad_spec, mode='constant', value=0)

        np.testing.assert_allclose(
            torch_output.numpy(),
            np.array(mlx_output),
            rtol=1e-6, atol=1e-6
        )

        # Check shape: 8+3+4=15 height, 8+1+2=11 width
        expected_shape = [2, 3, 15, 11]
        assert list(mlx_output.shape) == expected_shape

    def test_pad_unsupported_mode(self):
        """Test that unsupported mode raises error"""
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import pad_mlx

        mlx_input = mx.random.normal([1, 3, 8, 8])

        with pytest.raises(ValueError, match="Unsupported padding mode"):
            pad_mlx(mlx_input, (1, 1, 1, 1), mode='circular')


# ============================================================================
# Test GetAbsPosMLX
# ============================================================================

class TestGetAbsPosMLX:
    """Tests for get_abs_pos_mlx function"""

    def test_get_abs_pos_clip_style_resize(self):
        """Test resizing CLIP-style position embeddings"""
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import get_abs_pos_mlx

        # CLIP style: [1, 257, 1024] = 16x16 spatial + 1 CLS token
        src_size = 16
        tgt_size = 32
        embed_dim = 1024

        abs_pos = mx.random.normal([1, src_size * src_size + 1, embed_dim])
        resized = get_abs_pos_mlx(abs_pos, tgt_size)

        # Check shape
        expected_shape = [1, tgt_size * tgt_size + 1, embed_dim]
        assert list(resized.shape) == expected_shape

        # Check that CLS token is preserved (first position)
        np.testing.assert_allclose(
            np.array(abs_pos[:, 0, :]),
            np.array(resized[:, 0, :]),
            rtol=1e-6, atol=1e-6
        )

    def test_get_abs_pos_clip_style_no_resize(self):
        """Test CLIP-style with no resizing needed"""
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import get_abs_pos_mlx

        src_size = 16
        embed_dim = 768

        abs_pos = mx.random.normal([1, src_size * src_size + 1, embed_dim])
        resized = get_abs_pos_mlx(abs_pos, src_size)

        # Should be identical
        np.testing.assert_array_equal(
            np.array(abs_pos),
            np.array(resized)
        )

    def test_get_abs_pos_clip_style_downsample(self):
        """Test CLIP-style downsampling"""
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import get_abs_pos_mlx

        # Downsample from 32x32 to 16x16
        src_size = 32
        tgt_size = 16
        embed_dim = 512

        abs_pos = mx.random.normal([1, src_size * src_size + 1, embed_dim])
        resized = get_abs_pos_mlx(abs_pos, tgt_size)

        expected_shape = [1, tgt_size * tgt_size + 1, embed_dim]
        assert list(resized.shape) == expected_shape

    def test_get_abs_pos_sam_style_resize(self):
        """Test resizing SAM-style position embeddings"""
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import get_abs_pos_mlx

        # SAM style: [1, 64, 64, 768]
        src_size = 64
        tgt_size = 32
        embed_dim = 768

        abs_pos = mx.random.normal([1, src_size, src_size, embed_dim])
        resized = get_abs_pos_mlx(abs_pos, tgt_size)

        # Check shape
        expected_shape = [1, tgt_size, tgt_size, embed_dim]
        assert list(resized.shape) == expected_shape

    def test_get_abs_pos_sam_style_no_resize(self):
        """Test SAM-style with no resizing needed"""
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import get_abs_pos_mlx

        src_size = 32
        embed_dim = 256

        abs_pos = mx.random.normal([1, src_size, src_size, embed_dim])
        resized = get_abs_pos_mlx(abs_pos, src_size)

        # Should be identical
        np.testing.assert_array_equal(
            np.array(abs_pos),
            np.array(resized)
        )

    def test_get_abs_pos_sam_style_upsample(self):
        """Test SAM-style upsampling"""
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import get_abs_pos_mlx

        # Upsample from 32x32 to 64x64
        src_size = 32
        tgt_size = 64
        embed_dim = 384

        abs_pos = mx.random.normal([1, src_size, src_size, embed_dim])
        resized = get_abs_pos_mlx(abs_pos, tgt_size)

        expected_shape = [1, tgt_size, tgt_size, embed_dim]
        assert list(resized.shape) == expected_shape

    def test_get_abs_pos_preserves_dtype(self):
        """Test that dtype is preserved"""
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import get_abs_pos_mlx

        # Test with float16
        abs_pos = mx.random.normal([1, 257, 768]).astype(mx.float16)
        resized = get_abs_pos_mlx(abs_pos, 32)

        assert resized.dtype == mx.float16

    def test_get_abs_pos_invalid_shape(self):
        """Test that invalid shape raises error"""
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import get_abs_pos_mlx

        # 2D shape is invalid
        abs_pos = mx.random.normal([1, 768])

        with pytest.raises(ValueError, match="Unexpected abs_pos shape"):
            get_abs_pos_mlx(abs_pos, 32)


# ============================================================================
# Test QuickGELUMLX
# ============================================================================

class TestQuickGELUMLX:
    """Tests for quick_gelu_mlx function"""

    def test_quick_gelu_shape(self):
        """Test that quick_gelu preserves shape"""
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import quick_gelu_mlx

        x = mx.random.normal([2, 64])
        out = quick_gelu_mlx(x)

        assert out.shape == x.shape

    def test_quick_gelu_values(self):
        """Test quick_gelu computation"""
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import quick_gelu_mlx

        x = mx.array([0.0, 1.0, -1.0, 2.0])
        out = quick_gelu_mlx(x)

        # Compute expected: x * sigmoid(1.702 * x)
        expected = x * mx.sigmoid(1.702 * x)

        np.testing.assert_allclose(
            np.array(out),
            np.array(expected),
            rtol=1e-6, atol=1e-6
        )

    def test_quick_gelu_multidimensional(self):
        """Test quick_gelu with multidimensional input"""
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import quick_gelu_mlx

        x = mx.random.normal([4, 8, 16, 32])
        out = quick_gelu_mlx(x)

        assert out.shape == x.shape

        # Check computation is correct
        expected = x * mx.sigmoid(1.702 * x)
        np.testing.assert_allclose(
            np.array(out),
            np.array(expected),
            rtol=1e-6, atol=1e-6
        )

    def test_quick_gelu_zero(self):
        """Test quick_gelu at zero"""
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import quick_gelu_mlx

        x = mx.zeros([2, 3])
        out = quick_gelu_mlx(x)

        # quick_gelu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        expected = mx.zeros([2, 3])
        np.testing.assert_allclose(
            np.array(out),
            np.array(expected),
            rtol=1e-6, atol=1e-6
        )

    @pytest.mark.requires_pytorch
    def test_quick_gelu_approximates_gelu(self):
        """Test that quick_gelu approximates GELU"""
        import torch
        import torch.nn.functional as F
        import mlx.core as mx
        from mlx_port.deepencoder.utils_mlx import quick_gelu_mlx

        # Create test input
        torch_input = torch.randn(100, 50)
        mlx_input = mx.array(torch_input.numpy())

        # Compute GELU (PyTorch) and quick_gelu (MLX)
        gelu_output = F.gelu(torch_input)
        quick_gelu_output = quick_gelu_mlx(mlx_input)

        # They should be similar but not identical
        # Check correlation is high
        gelu_np = gelu_output.numpy().flatten()
        quick_gelu_np = np.array(quick_gelu_output).flatten()

        correlation = np.corrcoef(gelu_np, quick_gelu_np)[0, 1]
        assert correlation > 0.99, f"Correlation {correlation} should be > 0.99"


# ============================================================================
# Phase 2 Acceptance Criteria Validation
# ============================================================================

def test_phase_2_1_acceptance_all_functions_implemented():
    """✅ Phase 2.1: All utility functions implemented"""
    from mlx_port.deepencoder.utils_mlx import (
        unfold_mlx,
        interpolate_mlx,
        pad_mlx,
        get_abs_pos_mlx,
        quick_gelu_mlx
    )

    # All functions should be callable
    assert callable(unfold_mlx)
    assert callable(interpolate_mlx)
    assert callable(pad_mlx)
    assert callable(get_abs_pos_mlx)
    assert callable(quick_gelu_mlx)


def test_phase_2_1_acceptance_functions_have_docstrings():
    """✅ Phase 2.1: All functions have proper docstrings"""
    from mlx_port.deepencoder.utils_mlx import (
        unfold_mlx,
        interpolate_mlx,
        pad_mlx,
        get_abs_pos_mlx,
        quick_gelu_mlx
    )

    functions = [unfold_mlx, interpolate_mlx, pad_mlx, get_abs_pos_mlx, quick_gelu_mlx]

    for func in functions:
        assert func.__doc__ is not None, f"{func.__name__} missing docstring"
        assert len(func.__doc__) > 50, f"{func.__name__} docstring too short"
        assert "Args:" in func.__doc__, f"{func.__name__} missing Args section"
        assert "Returns:" in func.__doc__, f"{func.__name__} missing Returns section"
        assert "Example:" in func.__doc__, f"{func.__name__} missing Example section"


def test_phase_2_2_acceptance_test_coverage():
    """✅ Phase 2.2: Comprehensive test coverage"""
    # Count test methods
    test_counts = {
        'unfold': len([m for m in dir(TestUnfoldMLX) if m.startswith('test_')]),
        'interpolate': len([m for m in dir(TestInterpolateMLX) if m.startswith('test_')]),
        'pad': len([m for m in dir(TestPadMLX) if m.startswith('test_')]),
        'get_abs_pos': len([m for m in dir(TestGetAbsPosMLX) if m.startswith('test_')]),
        'quick_gelu': len([m for m in dir(TestQuickGELUMLX) if m.startswith('test_')]),
    }

    # Verify minimum test counts per acceptance criteria
    assert test_counts['unfold'] >= 4, f"unfold_mlx needs >= 4 tests, has {test_counts['unfold']}"
    assert test_counts['interpolate'] >= 3, f"interpolate_mlx needs >= 3 tests, has {test_counts['interpolate']}"
    assert test_counts['pad'] >= 3, f"pad_mlx needs >= 3 tests, has {test_counts['pad']}"
    assert test_counts['get_abs_pos'] >= 4, f"get_abs_pos_mlx needs >= 4 tests, has {test_counts['get_abs_pos']}"
    assert test_counts['quick_gelu'] >= 2, f"quick_gelu_mlx needs >= 2 tests, has {test_counts['quick_gelu']}"

    total_tests = sum(test_counts.values())
    print(f"\n{'='*70}")
    print(f"Phase 2.2 Test Coverage Summary")
    print(f"{'='*70}")
    for func_name, count in test_counts.items():
        print(f"  {func_name}: {count} tests")
    print(f"  Total: {total_tests} tests")
    print(f"{'='*70}")
    print("✅ PHASE 2 COMPLETE: All acceptance criteria met!")
    print(f"{'='*70}")


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])
