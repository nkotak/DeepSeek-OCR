"""SAM ViT-B Vision Encoder (MLX Implementation)

This module implements the Segment Anything Model (SAM) vision encoder using MLX.
It uses MLX's native scaled_dot_product_attention from mx.fast for efficient attention.

References:
    - Original SAM: https://github.com/facebookresearch/segment-anything
    - MLX SDPA: mx.fast.scaled_dot_product_attention (PR #2468)
"""
import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, Type
from functools import partial

from .utils_mlx import get_abs_pos_mlx, interpolate_mlx


class MLPBlock(nn.Module):
    """MLP block with GELU activation"""

    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ):
        """
        Args:
            embedding_dim: Input/output dimension
            mlp_dim: Hidden dimension
            act: Activation layer class
        """
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass: x -> Linear -> Activation -> Linear"""
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    """2D Layer Normalization for image features [B, C, H, W]"""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        """
        Args:
            num_channels: Number of channels
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.weight = mx.ones([num_channels])
        self.bias = mx.zeros([num_channels])
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        """
        Normalize across channel dimension

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Normalized tensor [B, C, H, W]
        """
        # Compute mean and variance across channel dimension
        u = mx.mean(x, axis=1, keepdims=True)
        s = mx.mean((x - u) ** 2, axis=1, keepdims=True)
        x = (x - u) / mx.sqrt(s + self.eps)

        # Apply learned affine transformation
        weight = self.weight.reshape([1, -1, 1, 1])
        bias = self.bias.reshape([1, -1, 1, 1])

        return weight * x + bias


class Attention(nn.Module):
    """Multi-head Attention with optional relative position embeddings

    Uses MLX's native scaled_dot_product_attention for efficient computation.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projection
            use_rel_pos: Whether to use relative positional embeddings
            rel_pos_zero_init: Whether to zero-initialize rel pos embeddings
            input_size: Input spatial size (H, W) if using relative position
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, \
                "Input size must be provided if using relative positional encoding."

            # Initialize relative positional embeddings
            # [2*H-1, head_dim], [2*W-1, head_dim]
            if rel_pos_zero_init:
                self.rel_pos_h = mx.zeros([2 * input_size[0] - 1, head_dim])
                self.rel_pos_w = mx.zeros([2 * input_size[1] - 1, head_dim])
            else:
                self.rel_pos_h = mx.random.normal([2 * input_size[0] - 1, head_dim]) * 0.02
                self.rel_pos_w = mx.random.normal([2 * input_size[1] - 1, head_dim]) * 0.02

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply multi-head attention

        Args:
            x: Input tensor of shape [B, H, W, C]

        Returns:
            Output tensor of shape [B, H, W, C]
        """
        B, H, W, _ = x.shape

        # QKV projection: [B, H, W, C] -> [B, H*W, 3, num_heads, head_dim]
        qkv = self.qkv(x).reshape([B, H * W, 3, self.num_heads, -1])

        # Split into q, k, v and transpose to [B, num_heads, H*W, head_dim]
        q, k, v = mx.split(qkv, 3, axis=2)
        q = mx.squeeze(q, axis=2).transpose([0, 2, 1, 3])  # [B, num_heads, H*W, head_dim]
        k = mx.squeeze(k, axis=2).transpose([0, 2, 1, 3])
        v = mx.squeeze(v, axis=2).transpose([0, 2, 1, 3])

        # Apply attention using MLX's native SDPA
        if self.use_rel_pos:
            # Compute relative position bias
            rel_h, rel_w = self._add_decomposed_rel_pos(q, (H, W), (H, W))

            # Combine height and width biases: [B, num_heads, H*W, H*W]
            # rel_h: [B, num_heads, H*W, H, 1]
            # rel_w: [B, num_heads, H*W, 1, W]
            # After transpose and sum: [B, num_heads, H*W, H*W]
            attn_bias = (rel_h + rel_w.transpose([0, 1, 2, 4, 3])).reshape(
                [B, self.num_heads, H * W, H * W]
            )

            # Scaled dot-product attention with bias using MLX native function
            x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=attn_bias)
        else:
            # Scaled dot-product attention without bias using MLX native function
            x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)

        # Reshape output: [B, num_heads, H*W, head_dim] -> [B, H, W, C]
        x = x.transpose([0, 2, 1, 3]).reshape([B, H, W, -1])

        # Output projection
        x = self.proj(x)

        return x

    def _add_decomposed_rel_pos(
        self,
        q: mx.array,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
    ) -> Tuple[mx.array, mx.array]:
        """
        Calculate decomposed Relative Positional Embeddings

        Args:
            q: Query tensor [B, num_heads, H*W, head_dim]
            q_size: Query spatial size (H, W)
            k_size: Key spatial size (H, W)

        Returns:
            Tuple of (rel_h, rel_w) bias tensors
        """
        q_h, q_w = q_size
        k_h, k_w = k_size

        # Get relative position embeddings
        Rh = self._get_rel_pos(q_h, k_h, self.rel_pos_h)  # [q_h, k_h, head_dim]
        Rw = self._get_rel_pos(q_w, k_w, self.rel_pos_w)  # [q_w, k_w, head_dim]

        B, num_heads, _, dim = q.shape

        # Reshape q: [B, num_heads, H*W, head_dim] -> [B*num_heads, H, W, head_dim]
        r_q = q.reshape([B * num_heads, q_h, q_w, dim])

        # Compute height contribution using einsum
        # [B*num_heads, H, W, C] @ [H, k_h, C] -> [B*num_heads, H, W, k_h]
        rel_h = mx.einsum('bhwc,hkc->bhwk', r_q, Rh)

        # Compute width contribution
        # [B*num_heads, H, W, C] @ [W, k_w, C] -> [B*num_heads, H, W, k_w]
        rel_w = mx.einsum('bhwc,wkc->bhwk', r_q, Rw)

        # Reshape for broadcasting:
        # rel_h: [B*num_heads, H*W, k_h, 1]
        # rel_w: [B*num_heads, H*W, 1, k_w]
        rel_h = rel_h.reshape([B * num_heads, q_h * q_w, k_h, 1])
        rel_w = rel_w.reshape([B * num_heads, q_h * q_w, 1, k_w])

        # Reshape back to separate batch and heads
        rel_h = rel_h.reshape([B, num_heads, q_h * q_w, k_h, 1])
        rel_w = rel_w.reshape([B, num_heads, q_h * q_w, 1, k_w])

        return rel_h, rel_w

    @staticmethod
    def _get_rel_pos(q_size: int, k_size: int, rel_pos: mx.array) -> mx.array:
        """
        Get relative positional embeddings according to query and key sizes

        Args:
            q_size: Size of query
            k_size: Size of key
            rel_pos: Relative position embeddings [L, C]

        Returns:
            Extracted position embeddings [q_size, k_size, C]
        """
        max_rel_dist = int(2 * max(q_size, k_size) - 1)

        # Interpolate if needed
        if rel_pos.shape[0] != max_rel_dist:
            dtype = rel_pos.dtype
            rel_pos = rel_pos.astype(mx.float32)

            # Reshape for interpolation: [L, C] -> [1, C, L, 1]
            rel_pos_resized = rel_pos.transpose([1, 0]).reshape([1, -1, rel_pos.shape[0], 1])

            # Interpolate to max_rel_dist
            rel_pos_resized = interpolate_mlx(
                rel_pos_resized,
                (max_rel_dist, 1),
                mode='linear'
            )

            # Reshape back: [1, C, max_rel_dist, 1] -> [max_rel_dist, C]
            rel_pos_resized = rel_pos_resized.reshape([-1, max_rel_dist]).transpose([1, 0])
            rel_pos_resized = rel_pos_resized.astype(dtype)
        else:
            rel_pos_resized = rel_pos

        # Scale coordinates with short length if shapes for q and k are different
        q_coords = mx.arange(q_size).reshape([-1, 1]) * max(k_size / q_size, 1.0)
        k_coords = mx.arange(k_size).reshape([1, -1]) * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

        return rel_pos_resized[relative_coords.astype(mx.int32)]


def window_partition(x: mx.array, window_size: int) -> Tuple[mx.array, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed

    Args:
        x: Input tensor [B, H, W, C]
        window_size: Window size

    Returns:
        windows: [B*num_windows, window_size, window_size, C]
        (Hp, Wp): Padded height and width before partition
    """
    B, H, W, C = x.shape

    # Calculate padding
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    # Apply padding if needed
    if pad_h > 0 or pad_w > 0:
        # Pad: [B, H, W, C] -> [B, H+pad_h, W+pad_w, C]
        x = mx.pad(x, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)])

    Hp, Wp = H + pad_h, W + pad_w

    # Reshape to windows
    # [B, Hp, Wp, C] -> [B, Hp//ws, ws, Wp//ws, ws, C]
    x = x.reshape([B, Hp // window_size, window_size, Wp // window_size, window_size, C])

    # Permute: [B, Hp//ws, Wp//ws, ws, ws, C]
    windows = x.transpose([0, 1, 3, 2, 4, 5])

    # Flatten: [B*num_windows, ws, ws, C]
    windows = windows.reshape([-1, window_size, window_size, C])

    return windows, (Hp, Wp)


def window_unpartition(
    windows: mx.array,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int]
) -> mx.array:
    """
    Window unpartition into original sequences and removing padding

    Args:
        windows: [B*num_windows, window_size, window_size, C]
        window_size: Window size
        pad_hw: Padded (H, W)
        hw: Original (H, W) before padding

    Returns:
        x: [B, H, W, C]
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    C = windows.shape[-1]

    # Reshape: [B*num_windows, ws, ws, C] -> [B, Hp//ws, Wp//ws, ws, ws, C]
    x = windows.reshape([B, Hp // window_size, Wp // window_size, window_size, window_size, C])

    # Permute: [B, Hp//ws, ws, Wp//ws, ws, C]
    x = x.transpose([0, 1, 3, 2, 4, 5])

    # Reshape: [B, Hp, Wp, C]
    x = x.reshape([B, Hp, Wp, C])

    # Remove padding if needed
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]

    return x


class Block(nn.Module):
    """Transformer block with support of window attention and residual propagation"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Args:
            dim: Number of input channels
            num_heads: Number of attention heads
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            qkv_bias: Whether to add bias in QKV projection
            norm_layer: Normalization layer
            act_layer: Activation layer
            use_rel_pos: Whether to use relative position embeddings
            rel_pos_zero_init: Whether to zero-initialize relative position
            window_size: Window size for window attention (0 = global)
            input_size: Input resolution for relative position
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(
            embedding_dim=dim,
            mlp_dim=int(dim * mlp_ratio),
            act=act_layer
        )

        self.window_size = window_size

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply transformer block with optional window attention

        Args:
            x: [B, H, W, C]

        Returns:
            [B, H, W, C]
        """
        shortcut = x
        x = self.norm1(x)

        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        # Attention
        x = self.attn(x)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        # Residual connection
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding using Conv2d"""

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        """
        Args:
            kernel_size: Patch size
            stride: Patch stride
            padding: Padding
            in_chans: Number of input channels
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Convert image to patches

        Args:
            x: [B, C, H, W]

        Returns:
            [B, H', W', embed_dim] where H'=H/stride, W'=W/stride
        """
        # Conv2d: [B, C, H, W] -> [B, embed_dim, H', W']
        x = self.proj(x)

        # Permute: [B, embed_dim, H', W'] -> [B, H', W', embed_dim]
        x = x.transpose([0, 2, 3, 1])

        return x


class ImageEncoderViT(nn.Module):
    """Vision Transformer encoder from SAM"""

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ):
        """
        Args:
            img_size: Input image size
            patch_size: Patch size
            in_chans: Number of input image channels
            embed_dim: Patch embedding dimension
            depth: Depth of ViT
            num_heads: Number of attention heads
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            out_chans: Output channels
            qkv_bias: Whether to use bias in QKV projection
            norm_layer: Normalization layer
            act_layer: Activation layer
            use_abs_pos: Whether to use absolute positional embeddings
            use_rel_pos: Whether to use relative positional embeddings
            rel_pos_zero_init: Whether to zero-initialize relative position
            window_size: Window size for window attention blocks
            global_attn_indexes: Indexes for blocks using global attention
        """
        super().__init__()
        self.img_size = img_size

        # Patch embedding
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # Absolute positional embedding
        self.pos_embed = None
        if use_abs_pos:
            self.pos_embed = mx.zeros([1, img_size // patch_size, img_size // patch_size, embed_dim])

        # Transformer blocks
        self.blocks = []
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        # Neck (output projection)
        self.neck = [
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        ]

        # Additional conv layers for deeper features
        self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Encode image to features

        Args:
            x: [B, 3, H, W] input image

        Returns:
            [B, 1024, H/64, W/64] encoded features
        """
        # Patch embedding: [B, 3, H, W] -> [B, H/16, W/16, embed_dim]
        x = self.patch_embed(x)

        # Add absolute position embedding if enabled
        if self.pos_embed is not None:
            x = x + get_abs_pos_mlx(self.pos_embed, x.shape[1])

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # Neck: [B, H/16, W/16, embed_dim] -> [B, embed_dim, H/16, W/16]
        x = x.transpose([0, 3, 1, 2])

        # Apply neck layers sequentially
        for layer in self.neck:
            x = layer(x)

        # Apply additional conv layers
        # [B, 256, H/16, W/16] -> [B, 512, H/32, W/32]
        conv2_output = self.net_2(x)

        # [B, 512, H/32, W/32] -> [B, 1024, H/64, W/64]
        conv3_output = self.net_3(conv2_output)

        return conv3_output


def build_sam_vit_b(checkpoint: Optional[str] = None) -> ImageEncoderViT:
    """
    Build SAM ViT-B vision encoder

    Configuration:
        - Image size: 1024x1024
        - Patch size: 16x16
        - Embedding dim: 768
        - Depth: 12 transformer blocks
        - Attention heads: 12
        - Window attention with 14x14 windows
        - Global attention at blocks 2, 5, 8, 11

    Args:
        checkpoint: Path to checkpoint file (not used in MLX, weights loaded separately)

    Returns:
        ImageEncoderViT model
    """
    return ImageEncoderViT(
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        out_chans=256,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_abs_pos=True,
        use_rel_pos=True,
        rel_pos_zero_init=True,
        window_size=14,
        global_attn_indexes=(2, 5, 8, 11),
    )
