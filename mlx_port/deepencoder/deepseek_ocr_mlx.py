"""DeepSeek-OCR Main Model (MLX Implementation)

This module implements the complete DeepSeek-OCR vision-language model using MLX.
It integrates SAM encoder, CLIP encoder, and MLP projector to process multi-scale images.

Key Features:
    - Multi-scale vision processing (global + local crops)
    - Dual encoder architecture (SAM + CLIP)
    - 2D tile layout with newline tokens
    - View separator for multi-view images
    - Dynamic resolution support

References:
    - Implementation based on: DeepSeek-OCR-vllm/deepseek_ocr.py
    - Uses MLX native operations throughout
"""
import mlx.core as mx
import mlx.nn as nn
from typing import Optional, List, Tuple, Dict, Any
import math

from .sam_vary_mlx import build_sam_vit_b
from .clip_mlx import build_clip_l
from .projector_mlx import build_linear_projector


class DeepSeekOCRVisionModel(nn.Module):
    """DeepSeek-OCR Vision Model for processing images into language model embeddings

    This model integrates:
        - SAM ViT-B encoder for spatial features
        - CLIP Large encoder for semantic features
        - MLP projector to map to language model dimension
        - Special tokens for 2D layout and view separation
    """

    def __init__(
        self,
        n_embed: int = 1280,
        tile_tag: str = "2D",
    ):
        """
        Args:
            n_embed: Language model embedding dimension (default: 1280)
            tile_tag: Tile layout format (default: "2D" for row-wise layout with newlines)
        """
        super().__init__()

        self.n_embed = n_embed
        self.tile_tag = tile_tag

        # Build vision encoders
        self.sam_model = build_sam_vit_b()
        self.vision_model = build_clip_l()

        # Build projector: SAM (1024) + CLIP (1024) = 2048 -> n_embed
        self.projector = build_linear_projector(input_dim=2048, n_embed=n_embed)

        # Special tokens for image formatting
        embed_std = 1.0 / math.sqrt(n_embed)
        if tile_tag == "2D":
            # <|view_separator|>: separates local crops from global view
            self.view_separator = mx.random.normal([n_embed]) * embed_std

            # <|\n|>: newline token added at end of each row in 2D grid
            self.image_newline = mx.random.normal([n_embed]) * embed_std
        else:
            raise ValueError(f"Only 2D tile_tag is supported, got: {tile_tag}")

    def process_single_image(
        self,
        image: mx.array,
        patches: Optional[mx.array] = None,
        crop_shape: Optional[Tuple[int, int]] = None
    ) -> mx.array:
        """
        Process a single image (with optional local crops) into vision embeddings

        Args:
            image: Global view image [B, 3, H, W]
            patches: Local crop patches [P, 3, H, W] or None
            crop_shape: (num_width_tiles, num_height_tiles) or None

        Returns:
            Vision embeddings [L, n_embed] ready for language model
        """
        # Process global view
        global_features = self._encode_image(image)  # [B, L, n_embed]

        # Check if we have local crops
        has_crops = patches is not None and mx.sum(patches).item() != 0

        if has_crops:
            # Process local patches
            local_features = self._encode_image(patches)  # [P, L, n_embed]

            # Format features with 2D layout
            formatted_features = self._format_features_2d(
                global_features,
                local_features,
                crop_shape
            )
        else:
            # Only global view
            formatted_features = self._format_global_only(global_features)

        return formatted_features

    def _encode_image(self, image: mx.array) -> mx.array:
        """
        Encode image through SAM + CLIP + Projector pipeline

        Args:
            image: Input image [B, 3, H, W]

        Returns:
            Encoded features [B, L, n_embed]
        """
        # SAM encoder: [B, 3, H, W] -> [B, 1024, H/64, W/64]
        sam_features = self.sam_model(image)

        # CLIP encoder: [B, 3, H, W] -> [B, L+1, 1024] (with CLS token)
        # Pass SAM features as patch_embeds hint (can be None)
        clip_features = self.vision_model(image, patch_embeds=sam_features)

        # Remove CLS token from CLIP: [B, L+1, 1024] -> [B, L, 1024]
        clip_no_cls = clip_features[:, 1:, :]

        # Flatten SAM features: [B, 1024, H, W] -> [B, H*W, 1024]
        B, C, H, W = sam_features.shape
        sam_flat = sam_features.reshape([B, C, -1]).transpose([0, 2, 1])

        # Concatenate: CLIP + SAM = [B, L, 2048]
        # Note: L should match between CLIP (without CLS) and SAM (flattened)
        combined = mx.concatenate([clip_no_cls, sam_flat], axis=-1)

        # Project to language model dimension: [B, L, 2048] -> [B, L, n_embed]
        projected = self.projector(combined)

        return projected

    def _format_features_2d(
        self,
        global_features: mx.array,
        local_features: mx.array,
        crop_shape: Tuple[int, int]
    ) -> mx.array:
        """
        Format global and local features with 2D layout

        Layout:
            [local_features_row_0 + newline]
            [local_features_row_1 + newline]
            ...
            [global_features_row_0 + newline]
            [global_features_row_1 + newline]
            ...
            [view_separator]

        Args:
            global_features: Global view features [1, L_global, n_embed]
            local_features: Local patch features [P, L_local, n_embed]
            crop_shape: (num_width_tiles, num_height_tiles)

        Returns:
            Formatted features [total_tokens, n_embed]
        """
        # Get dimensions
        _, hw_global, n_dim = global_features.shape
        h_global = w_global = int(hw_global ** 0.5)

        _, hw_local, _ = local_features.shape
        h_local = w_local = int(hw_local ** 0.5)

        num_width_tiles, num_height_tiles = crop_shape

        # Format global features: [1, L, C] -> [H, W, C]
        global_2d = global_features[0].reshape([h_global, w_global, n_dim])

        # Add newline token to each row: [H, W, C] -> [H, W+1, C]
        newline_global = mx.broadcast_to(
            self.image_newline.reshape([1, 1, -1]),
            [h_global, 1, n_dim]
        )
        global_with_newline = mx.concatenate([global_2d, newline_global], axis=1)

        # Flatten: [H, W+1, C] -> [H*(W+1), C]
        global_flat = global_with_newline.reshape([-1, n_dim])

        # Format local features: [P, L, C] where P = num_height_tiles * num_width_tiles
        # Reshape to: [num_height_tiles, num_width_tiles, h_local, w_local, C]
        local_4d = local_features.reshape([
            num_height_tiles, num_width_tiles, h_local, w_local, n_dim
        ])

        # Rearrange to: [num_height_tiles, h_local, num_width_tiles, w_local, C]
        # This groups tiles by rows and positions
        local_rearranged = local_4d.transpose([0, 2, 1, 3, 4])

        # Reshape to: [num_height_tiles*h_local, num_width_tiles*w_local, C]
        # This creates a large 2D grid of all local patches
        local_2d = local_rearranged.reshape([
            num_height_tiles * h_local,
            num_width_tiles * w_local,
            n_dim
        ])

        # Add newline token to each row: [H_local, W_local, C] -> [H_local, W_local+1, C]
        newline_local = mx.broadcast_to(
            self.image_newline.reshape([1, 1, -1]),
            [num_height_tiles * h_local, 1, n_dim]
        )
        local_with_newline = mx.concatenate([local_2d, newline_local], axis=1)

        # Flatten: [H_local, W_local+1, C] -> [H_local*(W_local+1), C]
        local_flat = local_with_newline.reshape([-1, n_dim])

        # Concatenate: [local_features, global_features, view_separator]
        view_sep = self.view_separator.reshape([1, -1])
        result = mx.concatenate([local_flat, global_flat, view_sep], axis=0)

        return result

    def _format_global_only(self, global_features: mx.array) -> mx.array:
        """
        Format global-only features with 2D layout (no local crops)

        Args:
            global_features: Global view features [1, L, n_embed]

        Returns:
            Formatted features [total_tokens, n_embed]
        """
        _, hw, n_dim = global_features.shape
        h = w = int(hw ** 0.5)

        # Reshape to 2D: [1, L, C] -> [H, W, C]
        global_2d = global_features[0].reshape([h, w, n_dim])

        # Add newline token to each row
        newline = mx.broadcast_to(
            self.image_newline.reshape([1, 1, -1]),
            [h, 1, n_dim]
        )
        global_with_newline = mx.concatenate([global_2d, newline], axis=1)

        # Flatten: [H, W+1, C] -> [H*(W+1), C]
        global_flat = global_with_newline.reshape([-1, n_dim])

        # Add view separator
        view_sep = self.view_separator.reshape([1, -1])
        result = mx.concatenate([global_flat, view_sep], axis=0)

        return result

    def __call__(
        self,
        pixel_values: List[mx.array],
        images_crop: List[Optional[mx.array]],
        images_spatial_crop: List[Tuple[int, int]]
    ) -> List[mx.array]:
        """
        Process multiple images with optional local crops

        Args:
            pixel_values: List of global view images, each [1, 3, H, W]
            images_crop: List of local patches, each [P, 3, H_crop, W_crop] or None
            images_spatial_crop: List of crop shapes, each (num_width_tiles, num_height_tiles)

        Returns:
            List of vision embeddings, each [total_tokens, n_embed]
        """
        results = []

        for idx in range(len(pixel_values)):
            image = pixel_values[idx]
            patches = images_crop[idx] if idx < len(images_crop) else None
            crop_shape = images_spatial_crop[idx] if idx < len(images_spatial_crop) else None

            # Process single image
            embeddings = self.process_single_image(image, patches, crop_shape)
            results.append(embeddings)

        return results


def build_deepseek_ocr_vision_model(n_embed: int = 1280) -> DeepSeekOCRVisionModel:
    """
    Build DeepSeek-OCR vision model

    Args:
        n_embed: Language model embedding dimension (default: 1280)

    Returns:
        DeepSeekOCRVisionModel instance
    """
    return DeepSeekOCRVisionModel(n_embed=n_embed, tile_tag="2D")


def merge_vision_embeddings(
    input_ids: mx.array,
    inputs_embeds: mx.array,
    vision_embeddings: List[mx.array],
    image_token_id: int
) -> mx.array:
    """
    Merge vision embeddings into text embeddings at image token positions

    Args:
        input_ids: Text token IDs [B, L]
        inputs_embeds: Text embeddings [B, L, D]
        vision_embeddings: List of vision embeddings, each [num_tokens, D]
        image_token_id: ID of <image> token

    Returns:
        Merged embeddings [B, L_new, D] where L_new includes vision tokens
    """
    if not vision_embeddings:
        return inputs_embeds

    batch_size = input_ids.shape[0]
    results = []

    for b in range(batch_size):
        # Get this batch's tokens and embeddings
        seq_ids = input_ids[b]
        seq_embeds = inputs_embeds[b]

        # Find image token positions
        image_positions = mx.where(seq_ids == image_token_id)[0]

        if len(image_positions) == 0:
            # No image tokens in this sequence
            results.append(seq_embeds)
            continue

        # Split sequence at image token positions and insert vision embeddings
        parts = []
        prev_pos = 0

        for idx, img_pos in enumerate(image_positions):
            img_pos_int = int(img_pos.item())

            # Add text tokens before this image
            if img_pos_int > prev_pos:
                parts.append(seq_embeds[prev_pos:img_pos_int])

            # Add vision embedding for this image
            if idx < len(vision_embeddings):
                parts.append(vision_embeddings[idx])

            prev_pos = img_pos_int + 1

        # Add remaining text tokens after last image
        if prev_pos < seq_embeds.shape[0]:
            parts.append(seq_embeds[prev_pos:])

        # Concatenate all parts
        merged_seq = mx.concatenate(parts, axis=0)
        results.append(merged_seq)

    # Note: Different sequences may have different lengths now
    # For batch processing, would need to pad to max length
    # For simplicity, returning list of variable-length sequences
    return results if batch_size > 1 else results[0]


def calculate_num_image_tokens(
    image_width: int,
    image_height: int,
    image_size: int = 1024,
    base_size: int = 1024,
    patch_size: int = 16,
    downsample_ratio: int = 4,
    crop_mode: bool = True
) -> int:
    """
    Calculate number of vision tokens for an image

    Args:
        image_width: Image width
        image_height: Image height
        image_size: Crop tile size (default: 1024)
        base_size: Global view size (default: 1024)
        patch_size: Vision encoder patch size (default: 16)
        downsample_ratio: Projector downsampling ratio (default: 4)
        crop_mode: Whether to use dynamic cropping

    Returns:
        Total number of vision tokens
    """
    # Calculate tokens per view
    h_global = w_global = math.ceil((base_size // patch_size) / downsample_ratio)
    h_local = w_local = math.ceil((image_size // patch_size) / downsample_ratio)

    # Global view tokens: h * (w + 1) where +1 is newline per row
    global_tokens = h_global * (w_global + 1)

    if crop_mode and (image_width > 640 or image_height > 640):
        # Estimate crop ratio (simplified)
        num_width_tiles = math.ceil(image_width / image_size)
        num_height_tiles = math.ceil(image_height / image_size)

        # Local view tokens
        local_tokens = (num_height_tiles * h_local) * (num_width_tiles * w_local + 1)
    else:
        local_tokens = 0

    # +1 for view separator
    return global_tokens + local_tokens + 1
