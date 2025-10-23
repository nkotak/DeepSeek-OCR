"""
Image preprocessing for DeepSeek-OCR MLX.

This module provides image preprocessing utilities including:
- Image transformations (normalization, conversion to MLX arrays)
- Multi-scale cropping (dynamic_preprocess)
- Aspect ratio selection
- Complete preprocessing pipeline (DeepseekOCRProcessor)

All operations use MLX native operations for Apple Silicon acceleration.
"""

import math
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int
) -> Tuple[int, int]:
    """
    Find the closest aspect ratio from target_ratios that matches the input aspect ratio.

    Args:
        aspect_ratio: Width / height ratio of input image
        target_ratios: List of (width_tiles, height_tiles) tuples
        width: Original image width
        height: Original image height
        image_size: Target tile size (e.g., 640, 1024)

    Returns:
        Best (width_tiles, height_tiles) tuple
    """
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)

        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            # Prefer larger grids for larger images
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def count_tiles(
    orig_width: int,
    orig_height: int,
    min_num: int = 1,
    max_num: int = 6,
    image_size: int = 640,
    use_thumbnail: bool = False
) -> Tuple[int, int]:
    """
    Calculate the optimal grid size (width_tiles, height_tiles) for cropping.

    Args:
        orig_width: Original image width
        orig_height: Original image height
        min_num: Minimum number of tiles
        max_num: Maximum number of tiles
        image_size: Target tile size (e.g., 640, 1024)
        use_thumbnail: Whether to add a thumbnail view

    Returns:
        (width_tiles, height_tiles) tuple

    Example:
        >>> count_tiles(1280, 640, image_size=640)
        (2, 1)  # 2x1 grid of 640x640 tiles
    """
    aspect_ratio = orig_width / orig_height

    # Generate all valid grid configurations
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    return target_aspect_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 6,
    image_size: int = 640,
    use_thumbnail: bool = False
) -> Tuple[List[Image.Image], Tuple[int, int]]:
    """
    Dynamically preprocess an image into multiple crops based on aspect ratio.

    This function:
    1. Determines the optimal grid size (e.g., 2x3) based on aspect ratio
    2. Resizes the image to fit the grid (e.g., 1280x1920 for 2x3 with 640px tiles)
    3. Crops the image into tiles (e.g., 6 crops of 640x640)
    4. Optionally adds a thumbnail view

    Args:
        image: PIL Image to preprocess
        min_num: Minimum number of crops
        max_num: Maximum number of crops
        image_size: Size of each crop (e.g., 640, 1024)
        use_thumbnail: Whether to add a downscaled thumbnail

    Returns:
        (processed_images, (width_tiles, height_tiles))
        - processed_images: List of PIL Images (crops + optional thumbnail)
        - (width_tiles, height_tiles): Grid dimensions

    Example:
        >>> image = Image.open("wide_image.png")  # 1920x1080
        >>> crops, grid = dynamic_preprocess(image, image_size=640)
        >>> len(crops)  # 3x2 grid = 6 crops
        6
        >>> grid
        (3, 2)
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Generate all valid grid configurations
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate target dimensions
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the image to fit the grid
    resized_img = image.resize((target_width, target_height))

    # Crop into tiles
    processed_images = []
    for i in range(blocks):
        col = i % (target_width // image_size)
        row = i // (target_width // image_size)
        box = (
            col * image_size,
            row * image_size,
            (col + 1) * image_size,
            (row + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks

    # Add thumbnail if requested
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images, target_aspect_ratio


class ImageTransform:
    """
    Transform PIL Images to normalized MLX arrays.

    Applies:
    1. Conversion to numpy array
    2. Normalization (mean/std)
    3. Conversion to MLX array

    Args:
        mean: RGB mean values for normalization
        std: RGB std values for normalization
        normalize: Whether to apply normalization

    Example:
        >>> transform = ImageTransform()
        >>> pil_img = Image.open("image.png")
        >>> mx_array = transform(pil_img)
        >>> mx_array.shape  # [3, H, W]
        (3, 1024, 1024)
    """

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize: bool = True
    ):
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)
        self.normalize = normalize

    def __call__(self, pil_img: Image.Image) -> mx.array:
        """
        Transform a PIL Image to a normalized MLX array.

        Args:
            pil_img: PIL Image (RGB)

        Returns:
            MLX array of shape [3, H, W] with values normalized
        """
        # Convert PIL to numpy: [H, W, 3] uint8
        img_np = np.array(pil_img, dtype=np.float32)

        # Transpose to [3, H, W] and normalize to [0, 1]
        img_np = img_np.transpose(2, 0, 1) / 255.0

        # Apply normalization: (x - mean) / std
        if self.normalize:
            img_np = (img_np - self.mean) / self.std

        # Convert to MLX array
        return mx.array(img_np, dtype=mx.float32)


@dataclass
class ProcessedImageData:
    """Container for preprocessed image data."""
    input_ids: mx.array  # Token IDs with image placeholders
    pixel_values: mx.array  # Global view: [B, 3, base_size, base_size]
    images_crop: mx.array  # Local crops: [B, num_crops, 3, image_size, image_size]
    images_seq_mask: mx.array  # Mask indicating image token positions
    images_spatial_crop: mx.array  # Crop grid dimensions: [B, 2] (width_tiles, height_tiles)
    num_image_tokens: List[int]  # Number of image tokens per image
    image_shapes: List[Tuple[int, int]]  # Original image shapes (width, height)


class DeepseekOCRProcessor:
    """
    Complete preprocessing pipeline for DeepSeek-OCR.

    This processor:
    1. Tokenizes text with <image> placeholders
    2. Preprocesses images (global view + local crops)
    3. Calculates image token sequences
    4. Returns all data needed for model forward pass

    Args:
        tokenizer: MLX tokenizer or compatible tokenizer
        image_size: Size for local crops (e.g., 640, 1024)
        base_size: Size for global view (e.g., 1024, 1280)
        patch_size: Vision encoder patch size (default: 16)
        downsample_ratio: Projector downsampling ratio (default: 4)
        image_mean: RGB mean for normalization
        image_std: RGB std for normalization
        normalize: Whether to normalize images
        image_token: Image placeholder token string
        min_crops: Minimum number of crops
        max_crops: Maximum number of crops

    Example:
        >>> processor = DeepseekOCRProcessor(tokenizer)
        >>> prompt = "Transcribe this image: <image>"
        >>> images = [Image.open("document.png")]
        >>> outputs = processor(prompt, images, cropping=True)
        >>> outputs.input_ids.shape
        [1, seq_len]
        >>> outputs.pixel_values.shape
        [1, 3, 1280, 1280]
    """

    def __init__(
        self,
        tokenizer: Any,
        image_size: int = 1024,
        base_size: int = 1280,
        patch_size: int = 16,
        downsample_ratio: int = 4,
        image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize: bool = True,
        image_token: str = "<image>",
        min_crops: int = 1,
        max_crops: int = 6,
    ):
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.base_size = base_size
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.image_mean = image_mean
        self.image_std = image_std
        self.normalize = normalize
        self.image_token = image_token
        self.min_crops = min_crops
        self.max_crops = max_crops

        # Image transformation
        self.image_transform = ImageTransform(
            mean=image_mean, std=image_std, normalize=normalize
        )

        # Get image token ID from tokenizer
        if hasattr(tokenizer, 'vocab') and image_token in tokenizer.vocab:
            self.image_token_id = tokenizer.vocab[image_token]
        elif hasattr(tokenizer, 'encode'):
            # Try encoding the token
            encoded = tokenizer.encode(image_token, add_special_tokens=False)
            self.image_token_id = encoded[0] if encoded else None
        else:
            self.image_token_id = None

        # Special token IDs
        self.bos_token_id = getattr(tokenizer, 'bos_token_id', 1)
        self.eos_token_id = getattr(tokenizer, 'eos_token_id', 2)
        self.pad_token_id = getattr(tokenizer, 'pad_token_id', 0)

    def encode_text(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text to token IDs."""
        if hasattr(self.tokenizer, 'encode'):
            return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        elif hasattr(self.tokenizer, '__call__'):
            result = self.tokenizer(text, add_special_tokens=add_special_tokens)
            return result['input_ids'] if isinstance(result, dict) else result
        else:
            raise ValueError("Tokenizer must have 'encode' or '__call__' method")

    def process_images_and_text(
        self,
        prompt: str,
        images: List[Image.Image],
        cropping: bool = True,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> ProcessedImageData:
        """
        Process images and text into model inputs.

        Args:
            prompt: Text prompt with <image> placeholders
            images: List of PIL Images
            cropping: Whether to use multi-scale cropping
            add_bos: Whether to add BOS token
            add_eos: Whether to add EOS token

        Returns:
            ProcessedImageData with all preprocessed inputs
        """
        # Verify image count matches placeholders
        assert prompt.count(self.image_token) == len(images), \
            f"Prompt has {prompt.count(self.image_token)} <image> tokens but {len(images)} images provided"

        # Split text by image token
        text_splits = prompt.split(self.image_token)

        # Initialize containers
        tokenized_str = []
        images_seq_mask = []
        images_list = []
        images_crop_list = []
        images_spatial_crop = []
        image_shapes = []
        num_image_tokens = []

        # Process each text-image pair
        for text_sep, image in zip(text_splits[:-1], images):
            # Encode text segment
            text_tokens = self.encode_text(text_sep, add_special_tokens=False)
            tokenized_str.extend(text_tokens)
            images_seq_mask.extend([False] * len(text_tokens))

            # Store original image shape
            image_shapes.append(image.size)

            # Determine cropping strategy
            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio = (1, 1)
                images_crop_raw = []
            else:
                if cropping:
                    images_crop_raw, crop_ratio = dynamic_preprocess(
                        image,
                        min_num=self.min_crops,
                        max_num=self.max_crops,
                        image_size=self.image_size
                    )
                else:
                    crop_ratio = (1, 1)
                    images_crop_raw = []

            # Process global view
            if self.image_size <= 640 and not cropping:
                image = image.resize((self.image_size, self.image_size))

            global_view = ImageOps.pad(
                image,
                (self.base_size, self.base_size),
                color=tuple(int(x * 255) for x in self.image_mean)
            )
            images_list.append(self.image_transform(global_view))

            # Record crop grid dimensions
            num_width_tiles, num_height_tiles = crop_ratio
            images_spatial_crop.append([num_width_tiles, num_height_tiles])

            # Process local crops
            if num_width_tiles > 1 or num_height_tiles > 1:
                for crop_img in images_crop_raw:
                    images_crop_list.append(self.image_transform(crop_img))

            # Calculate number of image tokens
            num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
            num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)

            # Token sequence: base grid + newlines + crops (if any) + view separator
            # Base: (num_queries_base + 1) * num_queries_base + 1
            tokenized_image = (
                [self.image_token_id] * num_queries_base + [self.image_token_id]
            ) * num_queries_base
            tokenized_image += [self.image_token_id]

            # Crops: (num_queries * width_tiles + 1) * (num_queries * height_tiles)
            if num_width_tiles > 1 or num_height_tiles > 1:
                tokenized_image += (
                    [self.image_token_id] * (num_queries * num_width_tiles) + [self.image_token_id]
                ) * (num_queries * num_height_tiles)

            tokenized_str.extend(tokenized_image)
            images_seq_mask.extend([True] * len(tokenized_image))
            num_image_tokens.append(len(tokenized_image))

        # Process final text segment
        final_text_tokens = self.encode_text(text_splits[-1], add_special_tokens=False)
        tokenized_str.extend(final_text_tokens)
        images_seq_mask.extend([False] * len(final_text_tokens))

        # Add BOS/EOS tokens
        if add_bos:
            tokenized_str = [self.bos_token_id] + tokenized_str
            images_seq_mask = [False] + images_seq_mask
        if add_eos:
            tokenized_str = tokenized_str + [self.eos_token_id]
            images_seq_mask = images_seq_mask + [False]

        # For inference, remove the final EOS token
        if add_eos:
            tokenized_str = tokenized_str[:-1]
            images_seq_mask = images_seq_mask[:-1]

        # Convert to MLX arrays
        input_ids = mx.array([tokenized_str], dtype=mx.int32)
        images_seq_mask_mx = mx.array([images_seq_mask], dtype=mx.bool_)

        # Stack images
        if len(images_list) == 0:
            pixel_values = mx.zeros((1, 3, self.base_size, self.base_size))
            images_spatial_crop_mx = mx.zeros((1, 2), dtype=mx.int32)
            images_crop = mx.zeros((1, 1, 3, self.image_size, self.image_size))
        else:
            pixel_values = mx.stack(images_list, axis=0)
            images_spatial_crop_mx = mx.array(images_spatial_crop, dtype=mx.int32)
            if images_crop_list:
                images_crop = mx.stack(images_crop_list, axis=0).reshape(
                    (1, -1, 3, self.image_size, self.image_size)
                )
            else:
                images_crop = mx.zeros((1, 1, 3, self.image_size, self.image_size))

        return ProcessedImageData(
            input_ids=input_ids,
            pixel_values=pixel_values,
            images_crop=images_crop,
            images_seq_mask=images_seq_mask_mx,
            images_spatial_crop=images_spatial_crop_mx,
            num_image_tokens=num_image_tokens,
            image_shapes=image_shapes,
        )

    def __call__(
        self,
        prompt: str,
        images: List[Image.Image],
        cropping: bool = True,
        **kwargs
    ) -> ProcessedImageData:
        """Process images and text."""
        return self.process_images_and_text(
            prompt=prompt,
            images=images,
            cropping=cropping,
            **kwargs
        )
