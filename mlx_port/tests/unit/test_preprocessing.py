"""
Unit tests for image preprocessing pipeline.

Tests:
- Image transformation (PIL to MLX array)
- Dynamic preprocessing (multi-scale cropping)
- Aspect ratio selection
- Complete preprocessing pipeline
"""

import unittest
from typing import List, Tuple
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    import mlx.core as mx
    import numpy as np
    from PIL import Image

    from preprocessing.image_processor_mlx import (
        ImageTransform,
        dynamic_preprocess,
        count_tiles,
        find_closest_aspect_ratio,
        DeepseekOCRProcessor,
    )

    MLX_AVAILABLE = True
except ImportError as e:
    MLX_AVAILABLE = False
    IMPORT_ERROR = str(e)


class DummyTokenizer:
    """Dummy tokenizer for testing."""

    def __init__(self):
        self.vocab = {"<image>": 128256, "<pad>": 0}
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        # Simple word-based encoding for testing
        words = text.split()
        token_ids = [hash(word) % 1000 + 100 for word in words]
        return token_ids


@unittest.skipUnless(MLX_AVAILABLE, f"MLX not available: {IMPORT_ERROR if not MLX_AVAILABLE else ''}")
class TestImageTransform(unittest.TestCase):
    """Test image transformation from PIL to MLX array."""

    def test_basic_transform(self):
        """Test basic image transformation."""
        transform = ImageTransform()

        # Create test image
        img = Image.new('RGB', (224, 224), color=(128, 128, 128))

        # Transform
        mx_img = transform(img)

        # Check shape
        self.assertEqual(mx_img.shape, (3, 224, 224))

        # Check dtype
        self.assertEqual(mx_img.dtype, mx.float32)

    def test_normalization(self):
        """Test image normalization."""
        transform = ImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)

        # Create white image (255, 255, 255)
        img = Image.new('RGB', (32, 32), color=(255, 255, 255))

        # Transform
        mx_img = transform(img)

        # White pixels should be normalized to (1.0 - 0.5) / 0.5 = 1.0
        expected_value = (1.0 - 0.5) / 0.5
        np.testing.assert_allclose(float(mx.mean(mx_img)), expected_value, rtol=1e-5)

    def test_no_normalization(self):
        """Test transformation without normalization."""
        transform = ImageTransform(normalize=False)

        # Create gray image (128, 128, 128)
        img = Image.new('RGB', (32, 32), color=(128, 128, 128))

        # Transform
        mx_img = transform(img)

        # Gray pixels should be 128/255 ≈ 0.502
        expected_value = 128.0 / 255.0
        np.testing.assert_allclose(float(mx.mean(mx_img)), expected_value, rtol=1e-3)


@unittest.skipUnless(MLX_AVAILABLE, f"MLX not available: {IMPORT_ERROR if not MLX_AVAILABLE else ''}")
class TestAspectRatioSelection(unittest.TestCase):
    """Test aspect ratio and tile count calculation."""

    def test_count_tiles_square(self):
        """Test tile counting for square image."""
        # 640x640 should use 1x1 grid
        tiles = count_tiles(640, 640, image_size=640)
        self.assertEqual(tiles, (1, 1))

    def test_count_tiles_wide(self):
        """Test tile counting for wide image."""
        # 1280x640 should use 2x1 grid
        tiles = count_tiles(1280, 640, image_size=640)
        self.assertEqual(tiles, (2, 1))

    def test_count_tiles_tall(self):
        """Test tile counting for tall image."""
        # 640x1280 should use 1x2 grid
        tiles = count_tiles(640, 1280, image_size=640)
        self.assertEqual(tiles, (1, 2))

    def test_count_tiles_large(self):
        """Test tile counting for large image."""
        # 1920x1280 should use 3x2 grid (6 tiles)
        tiles = count_tiles(1920, 1280, min_num=1, max_num=6, image_size=640)
        self.assertEqual(tiles[0] * tiles[1], 6)

    def test_find_closest_aspect_ratio(self):
        """Test finding closest aspect ratio."""
        target_ratios = [(1, 1), (2, 1), (1, 2), (2, 2)]

        # 16:9 aspect ratio should be closest to 2:1
        ratio = find_closest_aspect_ratio(
            aspect_ratio=16/9,
            target_ratios=target_ratios,
            width=1920,
            height=1080,
            image_size=640
        )
        self.assertEqual(ratio, (2, 1))


@unittest.skipUnless(MLX_AVAILABLE, f"MLX not available: {IMPORT_ERROR if not MLX_AVAILABLE else ''}")
class TestDynamicPreprocess(unittest.TestCase):
    """Test dynamic multi-scale preprocessing."""

    def test_single_crop(self):
        """Test preprocessing with single crop."""
        img = Image.new('RGB', (640, 640), color=(128, 128, 128))

        crops, grid = dynamic_preprocess(img, image_size=640, min_num=1, max_num=6)

        # Should return 1 crop for 640x640
        self.assertEqual(len(crops), 1)
        self.assertEqual(grid, (1, 1))

    def test_multiple_crops(self):
        """Test preprocessing with multiple crops."""
        img = Image.new('RGB', (1280, 640), color=(128, 128, 128))

        crops, grid = dynamic_preprocess(img, image_size=640, min_num=1, max_num=6)

        # Should return 2 crops for 1280x640 (2x1 grid)
        self.assertEqual(len(crops), 2)
        self.assertEqual(grid, (2, 1))

        # Each crop should be 640x640
        for crop in crops:
            self.assertEqual(crop.size, (640, 640))

    def test_thumbnail_mode(self):
        """Test preprocessing with thumbnail."""
        img = Image.new('RGB', (1280, 640), color=(128, 128, 128))

        crops, grid = dynamic_preprocess(img, image_size=640, use_thumbnail=True)

        # Should return crops + thumbnail
        self.assertEqual(len(crops), 3)  # 2 crops + 1 thumbnail


@unittest.skipUnless(MLX_AVAILABLE, f"MLX not available: {IMPORT_ERROR if not MLX_AVAILABLE else ''}")
class TestDeepseekOCRProcessor(unittest.TestCase):
    """Test complete preprocessing pipeline."""

    def setUp(self):
        """Set up test processor."""
        self.tokenizer = DummyTokenizer()
        self.processor = DeepseekOCRProcessor(
            self.tokenizer,
            image_size=1024,
            base_size=1280,
            image_token="<image>",
        )

    def test_processor_initialization(self):
        """Test processor initialization."""
        self.assertIsNotNone(self.processor.image_transform)
        self.assertEqual(self.processor.image_size, 1024)
        self.assertEqual(self.processor.base_size, 1280)

    def test_single_image_processing(self):
        """Test processing single image."""
        img = Image.new('RGB', (640, 640), color=(128, 128, 128))
        prompt = "Transcribe this image: <image>"

        outputs = self.processor(prompt, [img], cropping=False)

        # Check outputs
        self.assertIsNotNone(outputs.input_ids)
        self.assertIsNotNone(outputs.pixel_values)
        self.assertEqual(len(outputs.num_image_tokens), 1)

        # Check pixel values shape (global view)
        self.assertEqual(outputs.pixel_values.shape[1:], (3, 1280, 1280))

    def test_multi_image_processing(self):
        """Test processing multiple images."""
        img1 = Image.new('RGB', (640, 640), color=(128, 0, 0))
        img2 = Image.new('RGB', (640, 640), color=(0, 128, 0))
        prompt = "Compare <image> and <image>"

        outputs = self.processor(prompt, [img1, img2], cropping=False)

        # Check number of images
        self.assertEqual(len(outputs.num_image_tokens), 2)
        self.assertEqual(outputs.pixel_values.shape[0], 2)

    def test_multi_scale_processing(self):
        """Test multi-scale processing with crops."""
        img = Image.new('RGB', (1280, 640), color=(128, 128, 128))
        prompt = "OCR: <image>"

        outputs = self.processor(prompt, [img], cropping=True)

        # Should have crops
        self.assertTrue(outputs.images_crop.shape[1] > 1 or mx.sum(outputs.images_crop) != 0)

        # Check spatial crop dimensions
        self.assertEqual(outputs.images_spatial_crop.shape, (1, 2))

    def test_token_count_calculation(self):
        """Test image token count calculation."""
        img = Image.new('RGB', (640, 640), color=(128, 128, 128))
        prompt = "Text: <image>"

        outputs = self.processor(prompt, [img], cropping=False)

        # Number of tokens should match expected calculation
        # base_size=1280, patch_size=16, downsample_ratio=4
        # num_queries = ceil((1280 / 16) / 4) = ceil(80 / 4) = 20
        # Total tokens: (20 + 1) * 20 + 1 = 421
        expected_tokens = 421
        self.assertEqual(outputs.num_image_tokens[0], expected_tokens)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_tests()
