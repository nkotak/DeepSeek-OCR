"""
MLX utility functions for DeepSeek-OCR

This module provides MLX implementations of PyTorch operations used in DeepSeek-OCR.
All functions are designed to match PyTorch behavior for numerical accuracy.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Union, List, Optional


def unfold_mlx(
    x: mx.array,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    padding: int = 0
) -> mx.array:
    """
    MLX implementation of F.unfold for non-overlapping patches.

    This function extracts sliding local blocks from a batched input tensor.
    **Only supports non-overlapping patches (stride == kernel_size)**.

    Args:
        x: Input array of shape [B, C, H, W]
        kernel_size: Size of the sliding blocks (int or tuple of 2 ints)
        stride: Stride of the sliding blocks (int or tuple of 2 ints)
        padding: Implicit zero padding (must be 0, not implemented)

    Returns:
        Array of shape [B, C*kernel_h*kernel_w, num_patches]
        where num_patches = (H/kernel_h) * (W/kernel_w)

    Raises:
        NotImplementedError: If padding != 0 or stride != kernel_size
        ValueError: If dimensions not divisible by kernel_size

    Example:
        >>> x = mx.random.normal([2, 3, 4, 4])
        >>> out = unfold_mlx(x, kernel_size=2, stride=2)
        >>> out.shape
        [2, 12, 4]  # 12 = 3*2*2, 4 = (4/2)*(4/2)

    Mathematical Background:
        Input: x ∈ ℝ^(B×C×H×W)
        Operation: Extract k×k non-overlapping patches
        Output: y ∈ ℝ^(B×(C·k²)×((H/k)·(W/k)))

        Steps:
        1. Reshape: [B, C, H, W] → [B, C, H/k, k, W/k, k]
        2. Transpose: → [B, H/k, W/k, C, k, k]
        3. Flatten patches: → [B, (H/k)·(W/k), C·k²]
        4. Transpose to match PyTorch: → [B, C·k², (H/k)·(W/k)]
    """
    if padding != 0:
        raise NotImplementedError("unfold_mlx only supports padding=0")

    b, c, h, w = x.shape

    # Handle tuple or int kernel_size
    if isinstance(kernel_size, int):
        kh, kw = kernel_size, kernel_size
    else:
        kh, kw = kernel_size

    # Handle tuple or int stride
    if isinstance(stride, int):
        sh, sw = stride, stride
    else:
        sh, sw = stride

    # Verify non-overlapping (this is the simplification that makes it trivial)
    if sh != kh or sw != kw:
        raise NotImplementedError(
            f"unfold_mlx only supports non-overlapping patches (stride == kernel_size). "
            f"Got stride=({sh}, {sw}), kernel_size=({kh}, {kw})"
        )

    # Ensure dimensions are divisible
    if h % kh != 0 or w % kw != 0:
        raise ValueError(
            f"Height ({h}) and width ({w}) must be divisible by "
            f"kernel_size ({kh}, {kw})"
        )

    # Step 1: Reshape to separate patches
    # [B, C, H, W] → [B, C, H//kh, kh, W//kw, kw]
    x = x.reshape([b, c, h // kh, kh, w // kw, kw])

    # Step 2: Rearrange dimensions to group spatial patches
    # [B, C, H//kh, kh, W//kw, kw] → [B, H//kh, W//kw, C, kh, kw]
    x = x.transpose([0, 2, 4, 1, 3, 5])

    # Step 3: Flatten patches
    # [B, H//kh, W//kw, C, kh, kw] → [B, (H//kh)*(W//kw), C*kh*kw]
    num_patches_h = h // kh
    num_patches_w = w // kw
    x = x.reshape([b, num_patches_h * num_patches_w, c * kh * kw])

    # Step 4: Transpose to match PyTorch F.unfold output format
    # [B, num_patches, C*kh*kw] → [B, C*kh*kw, num_patches]
    x = x.transpose([0, 2, 1])

    return x


def interpolate_mlx(
    x: mx.array,
    size: Union[Tuple[int, int], List[int]],
    mode: str = 'bicubic',
    align_corners: bool = False,
    antialias: bool = True
) -> mx.array:
    """
    MLX implementation of F.interpolate for image resizing.

    Args:
        x: Input array of shape [B, C, H, W]
        size: Target size as (H, W)
        mode: Interpolation mode ('bilinear', 'bicubic', 'linear')
        align_corners: Whether to align corners (currently not used in MLX)
        antialias: Whether to use antialiasing

    Returns:
        Resized array of shape [B, C, size[0], size[1]]

    Example:
        >>> x = mx.random.normal([2, 3, 64, 64])
        >>> out = interpolate_mlx(x, (32, 32), mode='bicubic')
        >>> out.shape
        [2, 3, 32, 32]

    Note:
        MLX's image.resize operates on [H, W, C] format, so we need to
        transpose before and after resizing. Each image in the batch is
        processed separately.
    """
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise ValueError(f"size must be a tuple/list of 2 elements, got {size}")

    b, c, h, w = x.shape
    target_h, target_w = size

    # If already target size, return as-is
    if h == target_h and w == target_w:
        return x

    # MLX's image.resize operates on [H, W, C] format
    # We need to handle batch and channel dimensions

    # Approach: Process each image in batch separately
    resized_images = []

    for i in range(b):
        # Get single image: [C, H, W]
        img = x[i]

        # Transpose to [H, W, C] for mx.image.resize
        img = img.transpose([1, 2, 0])

        # Resize using MLX's image resize
        # Note: mx.image.resize expects size as [H, W]
        img_resized = mx.image.resize(
            img,
            [target_h, target_w],
            method=mode,
            antialias=antialias
        )

        # Transpose back to [C, H, W]
        img_resized = img_resized.transpose([2, 0, 1])

        resized_images.append(img_resized)

    # Stack batch: [B, C, H, W]
    result = mx.stack(resized_images, axis=0)

    return result


def pad_mlx(
    x: mx.array,
    pad: Tuple[int, ...],
    mode: str = 'constant',
    value: float = 0
) -> mx.array:
    """
    MLX implementation of F.pad.

    Args:
        x: Input array
        pad: Padding specification in PyTorch format (left, right, top, bottom, ...)
        mode: Padding mode ('constant', 'replicate'/'edge', 'reflect')
        value: Fill value for constant padding

    Returns:
        Padded array

    Note:
        PyTorch pad format: (left, right, top, bottom, front, back)
        MLX pad format: [(before, after), ...] for each dimension

    Example:
        >>> x = mx.random.normal([2, 3, 10, 10])
        >>> out = pad_mlx(x, (1, 1, 2, 2), mode='constant', value=0)
        >>> out.shape
        [2, 3, 14, 12]  # height: 10+2+2=14, width: 10+1+1=12
    """
    ndim = len(x.shape)

    # Convert PyTorch pad format to MLX format
    # PyTorch pad applies to last dimensions first
    mlx_pad = [(0, 0)] * ndim

    # Process pad values in pairs (before, after) for each dimension
    for i in range(len(pad) // 2):
        dim_idx = ndim - 1 - i  # Start from last dimension
        mlx_pad[dim_idx] = (pad[2 * i], pad[2 * i + 1])

    # Apply padding based on mode
    if mode == 'constant':
        return mx.pad(x, mlx_pad, constant_values=value)
    elif mode in ('replicate', 'edge'):
        return mx.pad(x, mlx_pad, mode='edge')
    elif mode == 'reflect':
        return mx.pad(x, mlx_pad, mode='reflect')
    else:
        raise ValueError(f"Unsupported padding mode: {mode}")


def get_abs_pos_mlx(abs_pos: mx.array, tgt_size: int) -> mx.array:
    """
    Resize absolute position embeddings to target size.

    Handles both CLIP-style (sequence with CLS token) and SAM-style (2D spatial)
    position embeddings.

    Args:
        abs_pos: Position embeddings
                 - CLIP style: [1, src_size²+1, C] (includes CLS token)
                 - SAM style: [1, src_size, src_size, C]
        tgt_size: Target spatial size

    Returns:
        Resized position embeddings in same format as input

    Example:
        >>> # CLIP-style
        >>> pos_embed = mx.random.normal([1, 257, 1024])  # 16x16 + 1 CLS
        >>> resized = get_abs_pos_mlx(pos_embed, 32)  # Resize to 32x32
        >>> resized.shape
        [1, 1025, 1024]  # 32x32 + 1 CLS

        >>> # SAM-style
        >>> pos_embed = mx.random.normal([1, 64, 64, 768])
        >>> resized = get_abs_pos_mlx(pos_embed, 32)
        >>> resized.shape
        [1, 32, 32, 768]

    Note:
        Uses bicubic interpolation for smooth resizing.
        Preserves dtype of input.
    """
    dtype = abs_pos.dtype

    # Handle CLIP-style: [1, seq_len, C] where seq_len = spatial_size² + 1
    if len(abs_pos.shape) == 3:
        # Calculate source spatial size (excluding CLS token)
        src_size = int((abs_pos.shape[1] - 1) ** 0.5)

        # Split CLS token and spatial embeddings
        cls_token = abs_pos[:, :1, :]  # [1, 1, C]
        pos_embed = abs_pos[:, 1:, :]   # [1, src_size², C]

        if src_size != tgt_size:
            # Reshape to 2D spatial: [1, src_size², C] → [1, src_size, src_size, C]
            c = pos_embed.shape[-1]
            pos_embed = pos_embed.reshape([1, src_size, src_size, c])

            # Transpose to [1, C, H, W] for resizing
            pos_embed = pos_embed.transpose([0, 3, 1, 2])

            # Resize
            pos_embed = pos_embed.astype(mx.float32)
            pos_embed = interpolate_mlx(
                pos_embed,
                (tgt_size, tgt_size),
                mode='bicubic',
                antialias=True
            )
            pos_embed = pos_embed.astype(dtype)

            # Transpose back and reshape: [1, C, H, W] → [1, tgt_size², C]
            pos_embed = pos_embed.transpose([0, 2, 3, 1])
            pos_embed = pos_embed.reshape([1, tgt_size * tgt_size, c])

            # Concatenate CLS token back
            return mx.concatenate([cls_token, pos_embed], axis=1)
        else:
            return abs_pos

    # Handle SAM-style: [1, H, W, C]
    elif len(abs_pos.shape) == 4:
        src_size = abs_pos.shape[1]

        if src_size != tgt_size:
            # Transpose to [1, C, H, W]
            abs_pos = abs_pos.transpose([0, 3, 1, 2])

            # Resize
            abs_pos = abs_pos.astype(mx.float32)
            abs_pos = interpolate_mlx(
                abs_pos,
                (tgt_size, tgt_size),
                mode='bicubic',
                antialias=True
            )
            abs_pos = abs_pos.astype(dtype)

            # Transpose back to [1, H, W, C]
            abs_pos = abs_pos.transpose([0, 2, 3, 1])

            return abs_pos
        else:
            return abs_pos
    else:
        raise ValueError(
            f"Unexpected abs_pos shape: {abs_pos.shape}. "
            f"Expected [1, seq_len, C] or [1, H, W, C]"
        )


def quick_gelu_mlx(x: mx.array) -> mx.array:
    """
    Quick GELU activation: x * sigmoid(1.702 * x)

    Used in CLIP encoder as a faster approximation of GELU.

    Args:
        x: Input array

    Returns:
        Activated array

    Example:
        >>> x = mx.random.normal([2, 64])
        >>> out = quick_gelu_mlx(x)
        >>> out.shape
        [2, 64]

    Note:
        This is an approximation of GELU that's faster to compute.
        The constant 1.702 is chosen to match GELU behavior closely.
    """
    return x * mx.sigmoid(1.702 * x)


# Export all functions
__all__ = [
    'unfold_mlx',
    'interpolate_mlx',
    'pad_mlx',
    'get_abs_pos_mlx',
    'quick_gelu_mlx',
]
