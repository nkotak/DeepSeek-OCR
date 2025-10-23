"""Unit tests for DeepSeek-OCR Main Model

Tests the complete DeepSeek-OCR vision model implementation.
Validates multi-scale vision processing, embedding merging, and integration.
"""
import pytest
import mlx.core as mx
import numpy as np
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deepencoder.deepseek_ocr_mlx import (
    DeepSeekOCRVisionModel,
    build_deepseek_ocr_vision_model,
    merge_vision_embeddings,
    calculate_num_image_tokens,
)


class TestDeepSeekOCRVisionModel:
    """Test main DeepSeek-OCR vision model"""

    @pytest.fixture
    def model(self):
        """Create DeepSeek-OCR vision model"""
        return build_deepseek_ocr_vision_model(n_embed=1280)

    def test_model_creation(self, model):
        """Test model can be created"""
        assert model is not None
        assert model.sam_model is not None
        assert model.vision_model is not None
        assert model.projector is not None
        assert model.image_newline is not None
        assert model.view_separator is not None
        print(f"✓ DeepSeek-OCR model created successfully")

    def test_special_tokens_shape(self, model):
        """Test special tokens have correct shape"""
        assert tuple(model.image_newline.shape) == (1280,)
        assert tuple(model.view_separator.shape) == (1280,)
        print(f"✓ Special tokens: image_newline={model.image_newline.shape}, "
              f"view_separator={model.view_separator.shape}")

    def test_encode_single_image(self, model):
        """Test encoding single image through full pipeline"""
        # Create test image: 1024x1024
        image = mx.random.normal([1, 3, 1024, 1024])

        # Encode through SAM + CLIP + Projector
        features = model._encode_image(image)

        # Expected output shape: [1, 256, 1280]
        # 256 = (1024/16/4)^2 = 16^2 tokens
        assert tuple(features.shape) == (1, 256, 1280)
        print(f"✓ Single image encoding: [1, 3, 1024, 1024] -> {tuple(features.shape)}")

    def test_encode_different_image_sizes(self, model):
        """Test encoding different image sizes"""
        test_sizes = [
            (512, 512, 64),    # 512x512 -> 8x8 = 64 tokens
            (1024, 1024, 256), # 1024x1024 -> 16x16 = 256 tokens
            (1280, 1280, 400), # 1280x1280 -> 20x20 = 400 tokens
        ]

        for h, w, expected_tokens in test_sizes:
            image = mx.random.normal([1, 3, h, w])
            features = model._encode_image(image)

            assert tuple(features.shape) == (1, expected_tokens, 1280), \
                f"Size {h}x{w}: expected {expected_tokens} tokens, got {features.shape[1]}"

        print(f"✓ Multiple image sizes: 512x512, 1024x1024, 1280x1280")

    def test_global_only_formatting(self, model):
        """Test formatting global-only features (no crops)"""
        # Create global features: 16x16 = 256 tokens
        global_features = mx.random.normal([1, 256, 1280])

        # Format with 2D layout
        formatted = model._format_global_only(global_features)

        # Expected: 16 rows * (16 + 1 newline) + 1 separator = 16*17 + 1 = 273
        expected_tokens = 16 * 17 + 1
        assert tuple(formatted.shape) == (expected_tokens, 1280)
        print(f"✓ Global-only formatting: 256 tokens -> {expected_tokens} tokens (with newlines + separator)")

    def test_2d_formatting_with_crops(self, model):
        """Test 2D formatting with local crops"""
        # Global features: 16x16 = 256 tokens
        global_features = mx.random.normal([1, 256, 1280])

        # Local features: 2x2 tiles, each 16x16 = 4 * 256 = 1024 tokens total
        local_features = mx.random.normal([4, 256, 1280])

        # Crop shape: 2 width tiles x 2 height tiles
        crop_shape = (2, 2)

        # Format
        formatted = model._format_features_2d(global_features, local_features, crop_shape)

        # Expected tokens:
        # Local: 32 rows (2 tiles * 16 rows) * (32 cols + 1 newline) = 32 * 33 = 1056
        # Global: 16 rows * (16 cols + 1 newline) = 16 * 17 = 272
        # Separator: 1
        # Total: 1056 + 272 + 1 = 1329
        expected_tokens = 32 * 33 + 16 * 17 + 1
        assert tuple(formatted.shape) == (expected_tokens, 1280)
        print(f"✓ 2D formatting with crops: "
              f"local=1024 + global=256 -> {expected_tokens} tokens (with newlines + separator)")

    def test_process_single_image_global_only(self, model):
        """Test processing single image without crops"""
        # Create test image
        image = mx.random.normal([1, 3, 1024, 1024])

        # Process without crops
        result = model.process_single_image(image, patches=None, crop_shape=None)

        # Expected: 16*17 + 1 = 273 tokens
        expected_tokens = 16 * 17 + 1
        assert tuple(result.shape) == (expected_tokens, 1280)
        print(f"✓ Process single image (global only): {tuple(result.shape)}")

    def test_process_single_image_with_crops(self, model):
        """Test processing single image with local crops"""
        # Global image: 1024x1024
        image = mx.random.normal([1, 3, 1024, 1024])

        # Local patches: 4 crops (2x2 grid), each 1024x1024
        patches = mx.random.normal([4, 3, 1024, 1024])

        # Crop shape
        crop_shape = (2, 2)

        # Process with crops
        result = model.process_single_image(image, patches, crop_shape)

        # Expected: 32*33 + 16*17 + 1 = 1329 tokens
        expected_tokens = 32 * 33 + 16 * 17 + 1
        assert tuple(result.shape) == (expected_tokens, 1280)
        print(f"✓ Process single image (with crops): {tuple(result.shape)}")

    def test_batch_processing(self, model):
        """Test processing multiple images"""
        # Create batch of images
        pixel_values = [
            mx.random.normal([1, 3, 1024, 1024]),
            mx.random.normal([1, 3, 1024, 1024])
        ]

        images_crop = [None, None]
        images_spatial_crop = [None, None]

        # Process batch
        results = model(pixel_values, images_crop, images_spatial_crop)

        assert len(results) == 2
        for result in results:
            expected_tokens = 16 * 17 + 1
            assert tuple(result.shape) == (expected_tokens, 1280)

        print(f"✓ Batch processing: 2 images")


class TestVisionEmbeddingMerging:
    """Test vision embedding merging into text embeddings"""

    def test_merge_single_image(self):
        """Test merging single vision embedding"""
        # Text: "Hello <image> world"
        # Token IDs: [1, 2, 100, 3, 4] where 100 is <image>
        input_ids = mx.array([[1, 2, 100, 3, 4]])

        # Text embeddings: [1, 5, 1280]
        inputs_embeds = mx.random.normal([1, 5, 1280])

        # Vision embedding: [273, 1280] (16x16 grid with newlines + separator)
        vision_embeddings = [mx.random.normal([273, 1280])]

        # Image token ID
        image_token_id = 100

        # Merge
        result = merge_vision_embeddings(
            input_ids, inputs_embeds, vision_embeddings, image_token_id
        )

        # Expected length: 2 (before) + 273 (vision) + 2 (after) = 277
        assert result.shape[0] == 277
        assert result.shape[1] == 1280
        print(f"✓ Merge single image: 5 text tokens + 273 vision tokens = {result.shape[0]} total")

    def test_merge_multiple_images(self):
        """Test merging multiple vision embeddings"""
        # Text: "<image> and <image>"
        # Token IDs: [100, 1, 100] where 100 is <image>
        input_ids = mx.array([[100, 1, 100]])

        # Text embeddings: [1, 3, 1280]
        inputs_embeds = mx.random.normal([1, 3, 1280])

        # Two vision embeddings
        vision_embeddings = [
            mx.random.normal([273, 1280]),  # First image
            mx.random.normal([273, 1280])   # Second image
        ]

        image_token_id = 100

        # Merge
        result = merge_vision_embeddings(
            input_ids, inputs_embeds, vision_embeddings, image_token_id
        )

        # Expected: 273 (img1) + 1 (text) + 273 (img2) = 547
        assert result.shape[0] == 547
        print(f"✓ Merge multiple images: 2 vision + 1 text token = {result.shape[0]} total")

    def test_merge_no_images(self):
        """Test merging with no vision embeddings"""
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        inputs_embeds = mx.random.normal([1, 5, 1280])
        vision_embeddings = []
        image_token_id = 100

        # Merge
        result = merge_vision_embeddings(
            input_ids, inputs_embeds, vision_embeddings, image_token_id
        )

        # Should return original embeddings
        assert mx.allclose(result, inputs_embeds[0], rtol=1e-6)
        print(f"✓ Merge with no images: returns original embeddings")


class TestTokenCalculation:
    """Test vision token calculation"""

    def test_calculate_tokens_small_image(self):
        """Test token calculation for small image (no crops)"""
        # 512x512 image
        num_tokens = calculate_num_image_tokens(
            image_width=512,
            image_height=512,
            crop_mode=False
        )

        # Expected: 16*17 + 1 = 273 (for 1024 base_size)
        assert num_tokens == 273
        print(f"✓ Small image (512x512, no crops): {num_tokens} tokens")

    def test_calculate_tokens_large_image_with_crops(self):
        """Test token calculation for large image with crops"""
        # 2048x2048 image (requires 2x2 crops of 1024)
        num_tokens = calculate_num_image_tokens(
            image_width=2048,
            image_height=2048,
            image_size=1024,
            crop_mode=True
        )

        # Expected: local (32*33) + global (16*17) + separator (1) = 1329
        expected = 32 * 33 + 16 * 17 + 1
        assert num_tokens == expected
        print(f"✓ Large image (2048x2048, 2x2 crops): {num_tokens} tokens")

    def test_calculate_tokens_different_sizes(self):
        """Test token calculation for different image sizes"""
        test_cases = [
            (640, 640, False, 273),     # Small, no crops
            (1024, 1024, False, 273),   # Medium, no crops
            (1280, 1280, False, 273),   # Large, no crops
        ]

        for width, height, crop_mode, expected in test_cases:
            num_tokens = calculate_num_image_tokens(
                image_width=width,
                image_height=height,
                crop_mode=crop_mode
            )
            assert num_tokens == expected, \
                f"Size {width}x{height}: expected {expected} tokens, got {num_tokens}"

        print(f"✓ Token calculation for multiple sizes")


class TestIntegration:
    """Integration tests for complete pipeline"""

    def test_end_to_end_single_image(self):
        """Test end-to-end pipeline for single image"""
        # Build model
        model = build_deepseek_ocr_vision_model(n_embed=1280)

        # Create input
        image = mx.random.normal([1, 3, 1024, 1024])

        # Process
        vision_embeddings = model.process_single_image(image, None, None)

        # Create text embeddings
        input_ids = mx.array([[1, 2, 100, 3]])  # 100 is <image>
        text_embeddings = mx.random.normal([1, 4, 1280])

        # Merge
        merged = merge_vision_embeddings(
            input_ids,
            text_embeddings,
            [vision_embeddings],
            image_token_id=100
        )

        # Expected: 2 (before <image>) + 273 (vision) + 1 (after) = 276
        assert merged.shape[0] == 276
        print(f"✓ End-to-end single image: {merged.shape[0]} total tokens")

    def test_end_to_end_multi_scale(self):
        """Test end-to-end pipeline for multi-scale image"""
        # Build model
        model = build_deepseek_ocr_vision_model(n_embed=1280)

        # Create inputs: global + 2x2 local crops
        global_image = mx.random.normal([1, 3, 1024, 1024])
        local_patches = mx.random.normal([4, 3, 1024, 1024])
        crop_shape = (2, 2)

        # Process
        vision_embeddings = model.process_single_image(
            global_image, local_patches, crop_shape
        )

        # Expected: 32*33 + 16*17 + 1 = 1329
        expected_tokens = 32 * 33 + 16 * 17 + 1
        assert tuple(vision_embeddings.shape) == (expected_tokens, 1280)
        print(f"✓ End-to-end multi-scale: {expected_tokens} vision tokens")

    def test_special_token_presence(self):
        """Test that special tokens are properly included"""
        model = build_deepseek_ocr_vision_model(n_embed=1280)

        # Process image
        image = mx.random.normal([1, 3, 1024, 1024])
        result = model.process_single_image(image, None, None)

        # Check that result includes newlines and separator
        # For 16x16 grid: should be 16*17 + 1 = 273 tokens
        # The +1 per row are newlines, and final +1 is separator
        assert result.shape[0] == 273
        print(f"✓ Special tokens included: newlines + view separator")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
