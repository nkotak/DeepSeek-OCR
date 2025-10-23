"""CLIP Vision Encoder (MLX Implementation)

This module implements the CLIP Large vision encoder using MLX.
It uses MLX's native scaled_dot_product_attention from mx.fast for efficient attention.

Key Differences from SAM:
    - Uses CLS token prepended to patch embeddings
    - No relative position embeddings (learned absolute positional embeddings)
    - Uses quick_gelu activation
    - Pre-LayerNorm architecture

References:
    - Original CLIP: https://github.com/openai/CLIP
    - MLX SDPA: mx.fast.scaled_dot_product_attention (PR #2468)
"""
import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
import math

from .utils_mlx import quick_gelu_mlx, interpolate_mlx


class CLIPVisionEmbeddings(nn.Module):
    """CLIP vision embeddings with patch embedding and class token"""

    def __init__(
        self,
        hidden_size: int = 1024,
        image_size: int = 224,
        patch_size: int = 14,
        num_channels: int = 3,
    ):
        """
        Args:
            hidden_size: Hidden dimension (embedding dimension)
            image_size: Input image size
            patch_size: Patch size
            num_channels: Number of input channels
        """
        super().__init__()
        self.embed_dim = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size

        # Class token embedding
        self.class_embedding = mx.random.normal([self.embed_dim]) * 0.02

        # Patch embedding using Conv2d
        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        # Position embeddings
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1  # +1 for CLS token

        # Initialize position embeddings
        self.position_embedding = mx.random.normal([1, self.num_positions, self.embed_dim]) * 0.02

    def __call__(self, pixel_values: mx.array, patch_embeds: Optional[mx.array] = None) -> mx.array:
        """
        Forward pass for vision embeddings

        Args:
            pixel_values: Input images [B, C, H, W]
            patch_embeds: Pre-computed patch embeddings (optional)

        Returns:
            Embeddings [B, num_positions, embed_dim]
        """
        batch_size = pixel_values.shape[0]

        # Compute patch embeddings if not provided
        if patch_embeds is not None:
            patch_embeds = patch_embeds
        else:
            # [B, C, H, W] -> [B, embed_dim, H/patch_size, W/patch_size]
            patch_embeds = self.patch_embedding(pixel_values)

        # Flatten and transpose: [B, embed_dim, H', W'] -> [B, H'*W', embed_dim]
        patch_embeds = patch_embeds.reshape([batch_size, self.embed_dim, -1]).transpose([0, 2, 1])

        # Prepend class token: [B, 1, embed_dim]
        class_embeds = mx.broadcast_to(
            self.class_embedding.reshape([1, 1, -1]),
            [batch_size, 1, self.embed_dim]
        )
        embeddings = mx.concatenate([class_embeds, patch_embeds], axis=1)

        # Add position embeddings with interpolation if needed
        embeddings = embeddings + self._get_abs_pos(self.position_embedding, embeddings.shape[1])

        return embeddings

    def _get_abs_pos(self, abs_pos: mx.array, tgt_size: int) -> mx.array:
        """
        Get absolute position embeddings with interpolation (CLIP-style with CLS token)

        Args:
            abs_pos: Position embeddings [1, L, C]
            tgt_size: Target sequence length (including CLS token)

        Returns:
            Interpolated position embeddings [1, tgt_size, C]
        """
        dim = abs_pos.shape[-1]

        # Split CLS token and spatial embeddings
        cls_token = abs_pos[:, :1, :]  # [1, 1, C]
        old_pos_embed = abs_pos[:, 1:, :]  # [1, L-1, C]

        src_size = int(math.sqrt(abs_pos.shape[1] - 1))
        tgt_size_spatial = int(math.sqrt(tgt_size - 1))

        if src_size != tgt_size_spatial:
            # Reshape for interpolation: [1, L-1, C] -> [1, C, src_size, src_size]
            old_pos_embed = old_pos_embed.reshape([1, src_size, src_size, dim])
            old_pos_embed = old_pos_embed.transpose([0, 3, 1, 2])

            # Interpolate
            dtype = abs_pos.dtype
            old_pos_embed = old_pos_embed.astype(mx.float32)
            new_pos_embed = interpolate_mlx(
                old_pos_embed,
                (tgt_size_spatial, tgt_size_spatial),
                mode='bicubic',
                antialias=True
            )
            new_pos_embed = new_pos_embed.astype(dtype)

            # Reshape back: [1, C, tgt_size, tgt_size] -> [1, tgt_size*tgt_size, C]
            new_pos_embed = new_pos_embed.transpose([0, 2, 3, 1])
            new_pos_embed = new_pos_embed.reshape([1, tgt_size_spatial * tgt_size_spatial, dim])

            # Concatenate CLS token back
            vision_pos_embed = mx.concatenate([cls_token, new_pos_embed], axis=1)
            return vision_pos_embed
        else:
            return abs_pos


class CLIPAttention(nn.Module):
    """Multi-head attention for CLIP using MLX native SDPA"""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.0,
    ):
        """
        Args:
            hidden_size: Hidden dimension
            num_attention_heads: Number of attention heads
            attention_dropout: Attention dropout probability (not used in inference)
        """
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=True)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        self.attn_drop = attention_dropout

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply multi-head attention

        Args:
            x: Input tensor [B, L, C]

        Returns:
            Output tensor [B, L, C]
        """
        bsz, seqlen, _ = x.shape

        # QKV projection: [B, L, C] -> [B, L, 3, num_heads, head_dim]
        xqkv = self.qkv_proj(x)
        xqkv = xqkv.reshape([bsz, seqlen, 3, self.num_heads, self.head_dim])

        # Split Q, K, V and transpose: [B, L, 3, num_heads, head_dim] -> 3 x [B, num_heads, L, head_dim]
        q, k, v = mx.split(xqkv, 3, axis=2)
        q = mx.squeeze(q, axis=2).transpose([0, 2, 1, 3])  # [B, num_heads, L, head_dim]
        k = mx.squeeze(k, axis=2).transpose([0, 2, 1, 3])
        v = mx.squeeze(v, axis=2).transpose([0, 2, 1, 3])

        # Apply scaled dot-product attention using MLX native SDPA
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)

        # Reshape: [B, num_heads, L, head_dim] -> [B, L, C]
        output = output.transpose([0, 2, 1, 3]).reshape([bsz, seqlen, -1])

        # Output projection
        output = self.out_proj(output)

        return output


class CLIPFeedForward(nn.Module):
    """Feed-forward network for CLIP with quick_gelu activation"""

    def __init__(self, dim: int, hidden_dim: int):
        """
        Args:
            dim: Input/output dimension
            hidden_dim: Hidden dimension
        """
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass: x -> Linear -> QuickGELU -> Linear

        Args:
            x: Input tensor [B, L, C]

        Returns:
            Output tensor [B, L, C]
        """
        return self.fc2(quick_gelu_mlx(self.fc1(x)))


class CLIPTransformerBlock(nn.Module):
    """Transformer block for CLIP with pre-LayerNorm"""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        ffn_hidden_size: int,
        layernorm_epsilon: float = 1e-5,
        attention_dropout: float = 0.0,
    ):
        """
        Args:
            hidden_size: Hidden dimension
            num_attention_heads: Number of attention heads
            ffn_hidden_size: FFN hidden dimension
            layernorm_epsilon: Layer norm epsilon
            attention_dropout: Attention dropout
        """
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=layernorm_epsilon)
        self.self_attn = CLIPAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
        )

        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=layernorm_epsilon)
        self.mlp = CLIPFeedForward(dim=hidden_size, hidden_dim=ffn_hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass with pre-LayerNorm and residual connections

        Args:
            x: Input tensor [B, L, C]

        Returns:
            Output tensor [B, L, C]
        """
        # Pre-norm attention
        residual = self.self_attn(self.layer_norm1(x))
        h = x + residual

        # Pre-norm FFN
        out = h + self.mlp(self.layer_norm2(h))

        return out


class CLIPTransformer(nn.Module):
    """Stack of CLIP transformer blocks"""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        ffn_hidden_size: int,
        layernorm_epsilon: float = 1e-5,
        attention_dropout: float = 0.0,
    ):
        """
        Args:
            num_layers: Number of transformer layers
            hidden_size: Hidden dimension
            num_attention_heads: Number of attention heads
            ffn_hidden_size: FFN hidden dimension
            layernorm_epsilon: Layer norm epsilon
            attention_dropout: Attention dropout
        """
        super().__init__()
        self.num_layers = num_layers

        self.layers = []
        for layer_id in range(num_layers):
            block = CLIPTransformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                ffn_hidden_size=ffn_hidden_size,
                layernorm_epsilon=layernorm_epsilon,
                attention_dropout=attention_dropout,
            )
            self.layers.append(block)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """
        Forward pass through all transformer layers

        Args:
            hidden_states: Input tensor [B, L, C]

        Returns:
            Output tensor [B, L, C]
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states


class CLIPVisionModel(nn.Module):
    """Complete CLIP vision encoder"""

    def __init__(
        self,
        hidden_size: int = 1024,
        num_layers: int = 24,
        num_attention_heads: int = 16,
        ffn_hidden_size: int = 4096,
        image_size: int = 224,
        patch_size: int = 14,
        layernorm_epsilon: float = 1e-5,
        pre_layernorm_epsilon: float = 1e-5,
        attention_dropout: float = 0.0,
        num_channels: int = 3,
    ):
        """
        Args:
            hidden_size: Hidden dimension
            num_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            ffn_hidden_size: FFN hidden dimension
            image_size: Input image size
            patch_size: Patch size
            layernorm_epsilon: Layer norm epsilon
            pre_layernorm_epsilon: Pre-layer norm epsilon
            attention_dropout: Attention dropout
            num_channels: Number of input channels
        """
        super().__init__()

        # Vision embeddings
        self.embeddings = CLIPVisionEmbeddings(
            hidden_size=hidden_size,
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
        )

        # Transformer
        self.transformer = CLIPTransformer(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            ffn_hidden_size=ffn_hidden_size,
            layernorm_epsilon=layernorm_epsilon,
            attention_dropout=attention_dropout,
        )

        # Pre-LayerNorm (applied after embeddings, before transformer)
        self.pre_layernorm = nn.LayerNorm(hidden_size, eps=pre_layernorm_epsilon)

    def __call__(
        self,
        pixel_values: mx.array,
        patch_embeds: Optional[mx.array] = None
    ) -> mx.array:
        """
        Forward pass through CLIP vision encoder

        Args:
            pixel_values: Input images [B, C, H, W]
            patch_embeds: Pre-computed patch embeddings (optional)

        Returns:
            Output features [B, L, C] where L = num_patches + 1 (includes CLS token)
        """
        # Embed patches and add position embeddings
        x = self.embeddings(pixel_values, patch_embeds)

        # Apply pre-LayerNorm
        hidden_states = self.pre_layernorm(x)

        # Apply transformer
        output = self.transformer(hidden_states)

        return output


def build_clip_l() -> CLIPVisionModel:
    """
    Build CLIP Large vision encoder

    Configuration:
        - Image size: 224x224
        - Patch size: 14x14 (16 patches per side)
        - Hidden size: 1024
        - Num layers: 24 transformer blocks
        - Attention heads: 16 (64 dims per head)
        - FFN hidden size: 4096
        - Activation: quick_gelu

    Returns:
        CLIPVisionModel instance
    """
    return CLIPVisionModel(
        hidden_size=1024,
        num_layers=24,
        num_attention_heads=16,
        ffn_hidden_size=4096,
        image_size=224,
        patch_size=14,
        layernorm_epsilon=1e-5,
        pre_layernorm_epsilon=1e-5,
        attention_dropout=0.0,
        num_channels=3,
    )
