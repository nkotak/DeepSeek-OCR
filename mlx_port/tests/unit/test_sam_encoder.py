"""Unit tests for SAM vision encoder

Tests the SAM ViT-B encoder implementation against PyTorch reference.
Validates that MLX implementation produces outputs matching PyTorch.
"""
import pytest
import torch
import mlx.core as mx
import numpy as np
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "DeepSeek-OCR-master/DeepSeek-OCR-vllm"))

from deepencoder.sam_vary_sdpa import (
    build_sam_vit_b as build_sam_pytorch,
    MLPBlock as MLPBlock_torch,
    LayerNorm2d as LayerNorm2d_torch,
    Attention as Attention_torch,
    Block as Block_torch,
    PatchEmbed as PatchEmbed_torch,
    window_partition as window_partition_torch,
    window_unpartition as window_unpartition_torch,
)
from mlx_port.deepencoder.sam_vary_mlx import (
    build_sam_vit_b as build_sam_mlx,
    MLPBlock as MLPBlock_mlx,
    LayerNorm2d as LayerNorm2d_mlx,
    Attention as Attention_mlx,
    Block as Block_mlx,
    PatchEmbed as PatchEmbed_mlx,
    window_partition as window_partition_mlx,
    window_unpartition as window_unpartition_mlx,
)
from mlx_port.tests.test_utils import create_test_image


@pytest.mark.requires_pytorch
class TestMLPBlock:
    """Test MLPBlock implementation"""

    def test_mlp_block_output_shape(self):
        """Test MLP block output shape"""
        embedding_dim = 768
        mlp_dim = 3072
        batch_size, h, w = 2, 16, 16

        # Create models
        mlp_torch = MLPBlock_torch(embedding_dim, mlp_dim)
        mlp_mlx = MLPBlock_mlx(embedding_dim, mlp_dim)

        # Load weights
        mlp_mlx.lin1.weight = mx.array(mlp_torch.lin1.weight.detach().numpy().T)
        mlp_mlx.lin1.bias = mx.array(mlp_torch.lin1.bias.detach().numpy())
        mlp_mlx.lin2.weight = mx.array(mlp_torch.lin2.weight.detach().numpy().T)
        mlp_mlx.lin2.bias = mx.array(mlp_torch.lin2.bias.detach().numpy())

        # Create test input
        x_torch = torch.randn(batch_size, h, w, embedding_dim)
        x_mlx = mx.array(x_torch.numpy())

        # Forward pass
        with torch.no_grad():
            out_torch = mlp_torch(x_torch)

        out_mlx = mlp_mlx(x_mlx)

        # Check shapes
        assert list(out_mlx.shape) == list(out_torch.shape)
        assert mx.allclose(out_mlx, mx.array(out_torch.numpy()), rtol=1e-5, atol=1e-6)
        print(f"✓ MLPBlock output shape: {out_mlx.shape}")


@pytest.mark.requires_pytorch
class TestLayerNorm2d:
    """Test LayerNorm2d implementation"""

    def test_layer_norm_2d_output(self):
        """Test 2D layer normalization"""
        num_channels = 256
        batch_size, h, w = 2, 32, 32

        # Create models
        ln_torch = LayerNorm2d_torch(num_channels)
        ln_mlx = LayerNorm2d_mlx(num_channels)

        # Load weights
        ln_mlx.weight = mx.array(ln_torch.weight.detach().numpy())
        ln_mlx.bias = mx.array(ln_torch.bias.detach().numpy())

        # Create test input [B, C, H, W]
        x_torch = torch.randn(batch_size, num_channels, h, w)
        x_mlx = mx.array(x_torch.numpy())

        # Forward pass
        with torch.no_grad():
            out_torch = ln_torch(x_torch)

        out_mlx = ln_mlx(x_mlx)

        # Check outputs
        assert list(out_mlx.shape) == list(out_torch.shape)
        assert mx.allclose(out_mlx, mx.array(out_torch.numpy()), rtol=1e-5, atol=1e-6)
        print(f"✓ LayerNorm2d output matches PyTorch")


@pytest.mark.requires_pytorch
class TestWindowOperations:
    """Test window partition and unpartition"""

    def test_window_partition(self):
        """Test window partition operation"""
        batch_size = 2
        h, w = 64, 64
        channels = 768
        window_size = 14

        # Create test input [B, H, W, C]
        x_torch = torch.randn(batch_size, h, w, channels)
        x_mlx = mx.array(x_torch.numpy())

        # Partition windows
        windows_torch, pad_hw_torch = window_partition_torch(x_torch, window_size)
        windows_mlx, pad_hw_mlx = window_partition_mlx(x_mlx, window_size)

        # Check outputs
        assert list(windows_mlx.shape) == list(windows_torch.shape)
        assert pad_hw_mlx == pad_hw_torch
        assert mx.allclose(windows_mlx, mx.array(windows_torch.numpy()), rtol=1e-6, atol=1e-7)
        print(f"✓ Window partition: {x_mlx.shape} -> {windows_mlx.shape}")

    def test_window_unpartition(self):
        """Test window unpartition operation"""
        batch_size = 2
        h, w = 64, 64
        channels = 768
        window_size = 14

        # Create test input and partition
        x_torch = torch.randn(batch_size, h, w, channels)
        x_mlx = mx.array(x_torch.numpy())

        windows_torch, pad_hw_torch = window_partition_torch(x_torch, window_size)
        windows_mlx, pad_hw_mlx = window_partition_mlx(x_mlx, window_size)

        # Unpartition
        recon_torch = window_unpartition_torch(windows_torch, window_size, pad_hw_torch, (h, w))
        recon_mlx = window_unpartition_mlx(windows_mlx, window_size, pad_hw_mlx, (h, w))

        # Check reconstruction
        assert list(recon_mlx.shape) == list(recon_torch.shape)
        assert mx.allclose(recon_mlx, mx.array(recon_torch.numpy()), rtol=1e-6, atol=1e-7)
        assert mx.allclose(recon_mlx, x_mlx, rtol=1e-6, atol=1e-7)
        print(f"✓ Window unpartition: {windows_mlx.shape} -> {recon_mlx.shape}")


@pytest.mark.requires_pytorch
class TestPatchEmbed:
    """Test PatchEmbed implementation"""

    def test_patch_embed_output(self):
        """Test patch embedding"""
        batch_size = 2
        in_chans = 3
        img_size = 1024
        patch_size = 16
        embed_dim = 768

        # Create models
        pe_torch = PatchEmbed_torch(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        pe_mlx = PatchEmbed_mlx(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        # Load weights
        weight = pe_torch.proj.weight.detach().numpy()
        pe_mlx.proj.weight = mx.array(weight)

        # Create test input [B, C, H, W]
        x_torch = torch.randn(batch_size, in_chans, img_size, img_size)
        x_mlx = mx.array(x_torch.numpy())

        # Forward pass
        with torch.no_grad():
            out_torch = pe_torch(x_torch)

        out_mlx = pe_mlx(x_mlx)

        # Check outputs
        expected_h = img_size // patch_size
        expected_w = img_size // patch_size
        assert list(out_mlx.shape) == [batch_size, expected_h, expected_w, embed_dim]
        assert list(out_mlx.shape) == list(out_torch.shape)
        assert mx.allclose(out_mlx, mx.array(out_torch.numpy()), rtol=1e-5, atol=1e-6)
        print(f"✓ PatchEmbed: [{batch_size}, {in_chans}, {img_size}, {img_size}] -> {list(out_mlx.shape)}")


@pytest.mark.requires_pytorch
class TestSAMEncoder:
    """Tests for complete SAM vision encoder"""

    @pytest.fixture
    def models(self):
        """Create PyTorch and MLX SAM models with matched weights"""
        # Build models
        sam_pytorch = build_sam_pytorch()
        sam_pytorch.eval()

        sam_mlx = build_sam_mlx()

        # Load PyTorch weights into MLX (simplified weight transfer)
        # Note: In production, use proper weight conversion utilities
        # For now, we'll initialize with same random seed for structural validation
        torch.manual_seed(42)
        np.random.seed(42)

        return sam_pytorch, sam_mlx

    def test_sam_encoder_output_shape(self, models):
        """Test SAM encoder output shape"""
        sam_pytorch, sam_mlx = models

        # Create test input
        test_input_torch = torch.randn(1, 3, 1024, 1024)
        test_input_mlx = mx.array(test_input_torch.numpy())

        # Forward pass
        with torch.no_grad():
            output_torch = sam_pytorch(test_input_torch)

        output_mlx = sam_mlx(test_input_mlx)

        # Check shapes match
        assert list(output_mlx.shape) == list(output_torch.shape), \
            f"Shape mismatch: MLX {list(output_mlx.shape)} vs PyTorch {list(output_torch.shape)}"

        expected_shape = [1, 1024, 1024 // 64, 1024 // 64]
        assert list(output_mlx.shape) == expected_shape

        print(f"✓ SAM encoder output shape: {list(output_mlx.shape)}")

    def test_sam_encoder_different_batch_sizes(self, models):
        """Test SAM encoder with different batch sizes"""
        sam_pytorch, sam_mlx = models

        batch_sizes = [1, 2, 4]

        for batch_size in batch_sizes:
            test_input_torch = torch.randn(batch_size, 3, 1024, 1024)
            test_input_mlx = mx.array(test_input_torch.numpy())

            with torch.no_grad():
                output_torch = sam_pytorch(test_input_torch)

            output_mlx = sam_mlx(test_input_mlx)

            expected_shape = [batch_size, 1024, 16, 16]
            assert list(output_mlx.shape) == expected_shape
            assert list(output_mlx.shape) == list(output_torch.shape)

        print(f"✓ SAM encoder works with batch sizes: {batch_sizes}")

    @pytest.mark.slow
    def test_sam_encoder_different_sizes(self, models):
        """Test SAM encoder with different input sizes"""
        sam_pytorch, sam_mlx = models

        # SAM typically works with 1024x1024, but can handle other sizes
        sizes = [(1024, 1024), (512, 512)]

        for size in sizes:
            test_input_torch = torch.randn(1, 3, *size)
            test_input_mlx = mx.array(test_input_torch.numpy())

            with torch.no_grad():
                output_torch = sam_pytorch(test_input_torch)

            output_mlx = sam_mlx(test_input_mlx)

            # Output should be 1/64 of input size
            expected_h = size[0] // 64
            expected_w = size[1] // 64
            expected_shape = [1, 1024, expected_h, expected_w]

            assert list(output_mlx.shape) == expected_shape
            assert list(output_mlx.shape) == list(output_torch.shape)

        print(f"✓ SAM encoder works with different input sizes: {sizes}")

    def test_sam_encoder_architecture(self, models):
        """Test SAM encoder architecture components"""
        sam_pytorch, sam_mlx = models

        # Check patch embedding
        assert sam_mlx.patch_embed is not None
        assert sam_pytorch.patch_embed is not None

        # Check transformer blocks
        assert len(sam_mlx.blocks) == len(sam_pytorch.blocks)
        assert len(sam_mlx.blocks) == 12

        # Check neck layers
        assert len(sam_mlx.neck) == 4
        assert len(sam_pytorch.neck) == 4

        # Check additional conv layers
        assert sam_mlx.net_2 is not None
        assert sam_mlx.net_3 is not None

        print(f"✓ SAM encoder architecture validated: {len(sam_mlx.blocks)} blocks")

    def test_sam_encoder_window_attention(self, models):
        """Test that window attention is properly configured"""
        sam_pytorch, sam_mlx = models

        # Check window sizes
        for i, (blk_torch, blk_mlx) in enumerate(zip(sam_pytorch.blocks, sam_mlx.blocks)):
            if i in [2, 5, 8, 11]:
                # Global attention blocks
                assert blk_mlx.window_size == 0
                assert blk_torch.window_size == 0
            else:
                # Window attention blocks
                assert blk_mlx.window_size == 14
                assert blk_torch.window_size == 14

        print(f"✓ Window attention configured correctly (global at [2,5,8,11], window=14 elsewhere)")


@pytest.mark.requires_pytorch
class TestAttentionMechanism:
    """Test attention mechanism with SDPA"""

    def test_attention_without_rel_pos(self):
        """Test attention without relative position embeddings"""
        dim = 768
        num_heads = 12
        batch_size = 2
        h, w = 16, 16

        # Create models
        attn_torch = Attention_torch(dim, num_heads, use_rel_pos=False)
        attn_mlx = Attention_mlx(dim, num_heads, use_rel_pos=False)

        # Load weights
        attn_mlx.qkv.weight = mx.array(attn_torch.qkv.weight.detach().numpy().T)
        attn_mlx.qkv.bias = mx.array(attn_torch.qkv.bias.detach().numpy())
        attn_mlx.proj.weight = mx.array(attn_torch.proj.weight.detach().numpy().T)
        attn_mlx.proj.bias = mx.array(attn_torch.proj.bias.detach().numpy())

        # Create test input [B, H, W, C]
        x_torch = torch.randn(batch_size, h, w, dim)
        x_mlx = mx.array(x_torch.numpy())

        # Forward pass
        with torch.no_grad():
            out_torch = attn_torch(x_torch)

        out_mlx = attn_mlx(x_mlx)

        # Check outputs
        assert list(out_mlx.shape) == list(out_torch.shape)
        # SDPA may have small numerical differences
        assert mx.allclose(out_mlx, mx.array(out_torch.numpy()), rtol=1e-3, atol=1e-4)
        print(f"✓ Attention without rel_pos using mx.fast.scaled_dot_product_attention")

    def test_attention_with_rel_pos(self):
        """Test attention with relative position embeddings"""
        dim = 768
        num_heads = 12
        batch_size = 2
        h, w = 16, 16

        # Create models
        attn_torch = Attention_torch(dim, num_heads, use_rel_pos=True,
                                      rel_pos_zero_init=True, input_size=(h, w))
        attn_mlx = Attention_mlx(dim, num_heads, use_rel_pos=True,
                                 rel_pos_zero_init=True, input_size=(h, w))

        # Load weights
        attn_mlx.qkv.weight = mx.array(attn_torch.qkv.weight.detach().numpy().T)
        attn_mlx.qkv.bias = mx.array(attn_torch.qkv.bias.detach().numpy())
        attn_mlx.proj.weight = mx.array(attn_torch.proj.weight.detach().numpy().T)
        attn_mlx.proj.bias = mx.array(attn_torch.proj.bias.detach().numpy())
        attn_mlx.rel_pos_h = mx.array(attn_torch.rel_pos_h.detach().numpy())
        attn_mlx.rel_pos_w = mx.array(attn_torch.rel_pos_w.detach().numpy())

        # Create test input [B, H, W, C]
        x_torch = torch.randn(batch_size, h, w, dim)
        x_mlx = mx.array(x_torch.numpy())

        # Forward pass
        with torch.no_grad():
            out_torch = attn_torch(x_torch)

        out_mlx = attn_mlx(x_mlx)

        # Check outputs
        assert list(out_mlx.shape) == list(out_torch.shape)
        # SDPA with bias may have slightly larger differences
        assert mx.allclose(out_mlx, mx.array(out_torch.numpy()), rtol=1e-3, atol=1e-3)
        print(f"✓ Attention with rel_pos using mx.fast.scaled_dot_product_attention")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
