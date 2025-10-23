"""Unit tests for MLP Projector

Tests the MLP projector implementation against PyTorch reference.
Validates that MLX implementation produces outputs matching PyTorch.
"""
import pytest
import torch
import torch.nn.functional as F
import mlx.core as mx
import numpy as np
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "DeepSeek-OCR-master/DeepSeek-OCR-vllm"))

from deepencoder.build_linear import MlpProjector as MlpProjector_torch
from mlx_port.deepencoder.projector_mlx import (
    MlpProjector as MlpProjector_mlx,
    build_linear_projector,
    build_downsample_projector,
)
from mlx_port.deepencoder.utils_mlx import unfold_mlx


@pytest.mark.requires_pytorch
class TestLinearProjector:
    """Test linear projector"""

    def test_linear_projector_output(self):
        """Test linear projector output matches PyTorch"""
        input_dim = 2048
        n_embed = 1280
        batch_size = 2
        seq_len = 256

        # Create config
        config_torch = type('Config', (), {
            'projector_type': 'linear',
            'input_dim': input_dim,
            'n_embed': n_embed,
        })()

        config_mlx = {
            "projector_type": "linear",
            "input_dim": input_dim,
            "n_embed": n_embed,
        }

        # Create projectors
        proj_torch = MlpProjector_torch(config_torch)
        proj_mlx = MlpProjector_mlx(config_mlx)

        # Load weights
        proj_mlx.layers.weight = mx.array(proj_torch.layers.weight.detach().numpy().T)
        proj_mlx.layers.bias = mx.array(proj_torch.layers.bias.detach().numpy())

        # Create test input
        x_torch = torch.randn(batch_size, seq_len, input_dim)
        x_mlx = mx.array(x_torch.numpy())

        # Forward pass
        with torch.no_grad():
            out_torch = proj_torch(x_torch)

        out_mlx = proj_mlx(x_mlx)

        # Check outputs
        assert list(out_mlx.shape) == list(out_torch.shape)
        assert list(out_mlx.shape) == [batch_size, seq_len, n_embed]
        assert mx.allclose(out_mlx, mx.array(out_torch.numpy()), rtol=1e-5, atol=1e-6)
        print(f"✓ Linear projector: {list(x_mlx.shape)} -> {list(out_mlx.shape)}")

    def test_build_linear_projector(self):
        """Test linear projector factory function"""
        proj = build_linear_projector(input_dim=2048, n_embed=1280)

        x = mx.random.normal([2, 256, 2048])
        output = proj(x)

        assert tuple(output.shape) == (2, 256, 1280)
        print(f"✓ build_linear_projector works correctly")


@pytest.mark.requires_pytorch
class TestMLPProjector:
    """Test MLP projector with GELU activation"""

    def test_mlp_gelu_projector(self):
        """Test MLP with GELU activation"""
        input_dim = 1024
        n_embed = 512
        depth = 2
        batch_size = 2
        seq_len = 64

        # Create config
        config_torch = type('Config', (), {
            'projector_type': 'mlp_gelu',
            'input_dim': input_dim,
            'n_embed': n_embed,
            'get': lambda self, k, d: depth if k == 'depth' else d,
        })()

        config_mlx = {
            "projector_type": "mlp_gelu",
            "input_dim": input_dim,
            "n_embed": n_embed,
            "depth": depth,
        }

        # Create projectors
        proj_torch = MlpProjector_torch(config_torch)
        proj_mlx = MlpProjector_mlx(config_mlx)

        # Load weights (simplified - just test structure)
        proj_torch.eval()

        # Create test input
        x_torch = torch.randn(batch_size, seq_len, input_dim)
        x_mlx = mx.array(x_torch.numpy())

        # Forward pass
        with torch.no_grad():
            out_torch = proj_torch(x_torch)

        out_mlx = proj_mlx(x_mlx)

        # Check output shape
        assert list(out_mlx.shape) == [batch_size, seq_len, n_embed]
        assert list(out_mlx.shape) == list(out_torch.shape)
        print(f"✓ MLP GELU projector: {list(x_mlx.shape)} -> {list(out_mlx.shape)}")


@pytest.mark.requires_pytorch
class TestDownsampleProjector:
    """Test downsampling projector with unfold"""

    def test_downsample_mlp_gelu(self):
        """Test downsampling projector"""
        input_dim = 512
        n_embed = 1024
        downsample_ratio = 2
        depth = 2
        mlp_ratio = 1

        # Input: 16x16 spatial grid
        batch_size = 2
        h = w = 16
        seq_len = h * w

        # Create config
        config_torch = type('Config', (), {
            'projector_type': 'downsample_mlp_gelu',
            'input_dim': input_dim,
            'n_embed': n_embed,
            'downsample_ratio': downsample_ratio,
            'get': lambda self, k, d: {
                'depth': depth,
                'mlp_ratio': mlp_ratio,
                'downsample_ratio': downsample_ratio,
            }.get(k, d),
        })()

        config_mlx = {
            "projector_type": "downsample_mlp_gelu",
            "input_dim": input_dim,
            "n_embed": n_embed,
            "downsample_ratio": downsample_ratio,
            "depth": depth,
            "mlp_ratio": mlp_ratio,
        }

        # Create projectors
        proj_torch = MlpProjector_torch(config_torch)
        proj_mlx = MlpProjector_mlx(config_mlx)

        # Load weights (simplified)
        proj_torch.eval()

        # Create test input [B, H*W, C]
        x_torch = torch.randn(batch_size, seq_len, input_dim)
        x_mlx = mx.array(x_torch.numpy())

        # Forward pass
        with torch.no_grad():
            out_torch = proj_torch(x_torch)

        out_mlx = proj_mlx(x_mlx)

        # After downsampling: 16x16 -> 8x8 = 64 tokens
        expected_seq_len = (h // downsample_ratio) * (w // downsample_ratio)
        assert list(out_mlx.shape) == [batch_size, expected_seq_len, n_embed]
        assert list(out_mlx.shape) == list(out_torch.shape)
        print(f"✓ Downsample projector: [{batch_size}, {seq_len}, {input_dim}] -> {list(out_mlx.shape)}")

    def test_downsample_with_padding(self):
        """Test downsampling with padding for non-divisible sizes"""
        input_dim = 256
        n_embed = 512
        downsample_ratio = 2

        # Input: 15x15 spatial grid (not divisible by 2)
        batch_size = 1
        h = w = 15
        seq_len = h * w

        config_mlx = {
            "projector_type": "downsample_mlx_gelu",
            "input_dim": input_dim,
            "n_embed": n_embed,
            "downsample_ratio": downsample_ratio,
            "depth": 1,
            "mlp_ratio": 1,
        }

        proj_mlx = MlpProjector_mlx(config_mlx)

        # Create test input
        x_mlx = mx.random.normal([batch_size, seq_len, input_dim])

        # Forward pass
        out_mlx = proj_mlx(x_mlx)

        # After padding: 15 -> 16, then downsample 16/2 = 8
        expected_seq_len = 8 * 8
        assert tuple(out_mlx.shape) == (batch_size, expected_seq_len, n_embed)
        print(f"✓ Downsample with padding: 15x15 -> 16x16 -> 8x8")

    def test_build_downsample_projector(self):
        """Test downsample projector factory function"""
        proj = build_downsample_projector(
            input_dim=512,
            n_embed=1024,
            downsample_ratio=2,
            depth=2,
            mlp_ratio=1,
            use_norm=False
        )

        # Test with 16x16 input
        x = mx.random.normal([2, 256, 512])
        output = proj(x)

        # After 2x downsampling: 256 -> 64 tokens
        assert tuple(output.shape) == (2, 64, 1024)
        print(f"✓ build_downsample_projector works correctly")


@pytest.mark.requires_pytorch
class TestNormLayerDownsampleProjector:
    """Test downsample projector with LayerNorm"""

    def test_normlayer_downsample_mlp_gelu(self):
        """Test downsample projector with LayerNorm"""
        input_dim = 384
        n_embed = 768
        downsample_ratio = 2
        depth = 2
        mlp_ratio = 2

        batch_size = 2
        h = w = 16
        seq_len = h * w

        config_mlx = {
            "projector_type": "normlayer_downsample_mlp_gelu",
            "input_dim": input_dim,
            "n_embed": n_embed,
            "downsample_ratio": downsample_ratio,
            "depth": depth,
            "mlp_ratio": mlp_ratio,
        }

        proj_mlx = MlpProjector_mlx(config_mlx)

        # Create test input
        x_mlx = mx.random.normal([batch_size, seq_len, input_dim])

        # Forward pass
        out_mlx = proj_mlx(x_mlx)

        # After downsampling: 16x16 -> 8x8 = 64 tokens
        expected_seq_len = (h // downsample_ratio) * (w // downsample_ratio)
        assert tuple(out_mlx.shape) == (batch_size, expected_seq_len, n_embed)
        print(f"✓ NormLayer downsample projector: {list(out_mlx.shape)}")


@pytest.mark.requires_pytorch
class TestIdentityProjector:
    """Test identity projector"""

    def test_identity_projector(self):
        """Test identity projector (pass-through)"""
        config_mlx = {
            "projector_type": "identity",
            "input_dim": 1024,
            "n_embed": 1024,
        }

        proj_mlx = MlpProjector_mlx(config_mlx)

        # Create test input
        x_mlx = mx.random.normal([2, 256, 1024])

        # Forward pass
        out_mlx = proj_mlx(x_mlx)

        # Should be identical (pass-through)
        assert mx.allclose(out_mlx, x_mlx, rtol=1e-7, atol=1e-8)
        print(f"✓ Identity projector (pass-through)")


@pytest.mark.requires_pytorch
class TestUnfoldOperation:
    """Test unfold operation used for downsampling"""

    def test_unfold_2x2(self):
        """Test 2x2 unfold operation"""
        batch_size = 2
        channels = 3
        h = w = 4

        # Create test input [B, C, H, W]
        x_torch = torch.randn(batch_size, channels, h, w)
        x_mlx = mx.array(x_torch.numpy())

        # PyTorch unfold
        out_torch = F.unfold(x_torch, kernel_size=2, stride=2, padding=0)

        # MLX unfold
        out_mlx = unfold_mlx(x_mlx, kernel_size=2, stride=2, padding=0)

        # Check shapes: [B, C*k*k, num_patches]
        # 4x4 with 2x2 kernel, stride 2 -> 2x2 patches = 4 patches
        # channels: 3 * 2 * 2 = 12
        expected_shape = [batch_size, channels * 2 * 2, 2 * 2]
        assert list(out_mlx.shape) == expected_shape
        assert list(out_mlx.shape) == list(out_torch.shape)
        assert mx.allclose(out_mlx, mx.array(out_torch.numpy()), rtol=1e-5, atol=1e-6)
        print(f"✓ Unfold 2x2: {list(x_mlx.shape)} -> {list(out_mlx.shape)}")

    def test_unfold_with_different_kernel_sizes(self):
        """Test unfold with different kernel sizes"""
        batch_size = 1
        channels = 64
        h = w = 16

        x_torch = torch.randn(batch_size, channels, h, w)
        x_mlx = mx.array(x_torch.numpy())

        # Test different kernel sizes
        for kernel_size in [2, 4]:
            stride = kernel_size

            out_torch = F.unfold(x_torch, kernel_size=kernel_size, stride=stride, padding=0)
            out_mlx = unfold_mlx(x_mlx, kernel_size=kernel_size, stride=stride, padding=0)

            assert list(out_mlx.shape) == list(out_torch.shape)
            assert mx.allclose(out_mlx, mx.array(out_torch.numpy()), rtol=1e-5, atol=1e-6)

        print(f"✓ Unfold with different kernel sizes: [2, 4]")


@pytest.mark.requires_pytorch
class TestProjectorIntegration:
    """Test projector integration with vision encoders"""

    def test_sam_clip_to_projector(self):
        """Test typical SAM+CLIP to projector pipeline"""
        batch_size = 2

        # Simulate SAM features: [B, 1024, H/64, W/64]
        # For 1024x1024 input: [B, 1024, 16, 16]
        sam_features = mx.random.normal([batch_size, 1024, 16, 16])

        # Flatten SAM: [B, 1024, 16, 16] -> [B, 256, 1024]
        sam_flat = sam_features.reshape([batch_size, 1024, -1]).transpose([0, 2, 1])

        # Simulate CLIP features: [B, 257, 1024] (with CLS token)
        clip_features = mx.random.normal([batch_size, 257, 1024])

        # Remove CLS token: [B, 257, 1024] -> [B, 256, 1024]
        clip_no_cls = clip_features[:, 1:, :]

        # Concatenate: [B, 256, 1024] + [B, 256, 1024] -> [B, 256, 2048]
        combined = mx.concatenate([clip_no_cls, sam_flat], axis=-1)

        # Project: 2048 -> 1280
        proj = build_linear_projector(input_dim=2048, n_embed=1280)
        output = proj(combined)

        # Check output shape
        assert tuple(output.shape) == (batch_size, 256, 1280)
        print(f"✓ SAM+CLIP integration: [B, 256, 2048] -> [B, 256, 1280]")

    def test_projector_types_output_shapes(self):
        """Test that all projector types produce correct output shapes"""
        batch_size = 2
        seq_len = 64
        input_dim = 512
        n_embed = 1024

        projector_configs = [
            {"projector_type": "identity", "input_dim": input_dim, "n_embed": input_dim},
            {"projector_type": "linear", "input_dim": input_dim, "n_embed": n_embed},
            {"projector_type": "mlp_gelu", "input_dim": input_dim, "n_embed": n_embed, "depth": 2},
        ]

        for config in projector_configs:
            proj = MlpProjector_mlx(config)
            x = mx.random.normal([batch_size, seq_len, input_dim])
            output = proj(x)

            expected_shape = (batch_size, seq_len, config["n_embed"])
            assert tuple(output.shape) == expected_shape

        print(f"✓ All projector types produce correct shapes")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
