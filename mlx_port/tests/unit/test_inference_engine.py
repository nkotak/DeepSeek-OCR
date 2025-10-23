"""
Unit tests for inference engine (generation and pipeline).

Tests:
- DeepseekOCRForCausalLM model
- Vision processing and embedding merging
- Text generation (greedy and sampling)
- Streaming generation
- Complete inference pipeline
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    import mlx.core as mx
    import mlx.nn as nn
    from PIL import Image

    from models.deepseek_ocr_causal_lm_mlx import (
        DeepseekOCRForCausalLM,
        DeepseekOCRConfig,
        build_deepseek_ocr_model,
    )
    from inference.generation_mlx import (
        SamplingConfig,
        apply_temperature,
        sample_token,
    )
    from inference.pipeline_mlx import DeepSeekOCRPipeline
    from preprocessing.image_processor_mlx import DeepseekOCRProcessor

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

    def encode(self, text: str, add_special_tokens: bool = False) -> list:
        words = text.split()
        return [hash(word) % 1000 + 100 for word in words]

    def decode(self, token_ids: list, skip_special_tokens: bool = True) -> str:
        return " ".join([f"token_{id}" for id in token_ids[:5]])  # Simplified


@unittest.skipUnless(MLX_AVAILABLE, f"MLX not available: {IMPORT_ERROR if not MLX_AVAILABLE else ''}")
class TestDeepseekOCRConfig(unittest.TestCase):
    """Test model configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = DeepseekOCRConfig()

        self.assertEqual(config.tile_tag, "2D")
        self.assertEqual(config.projector_type, "linear")
        self.assertEqual(config.n_embed, 1280)

    def test_custom_config(self):
        """Test custom configuration."""
        config = DeepseekOCRConfig(
            n_embed=2048,
            projector_type="mlp_gelu",
            image_token_id=999,
        )

        self.assertEqual(config.n_embed, 2048)
        self.assertEqual(config.projector_type, "mlp_gelu")
        self.assertEqual(config.image_token_id, 999)

    def test_invalid_tile_tag(self):
        """Test that invalid tile_tag raises error."""
        with self.assertRaises(ValueError):
            DeepseekOCRConfig(tile_tag="1D")


@unittest.skipUnless(MLX_AVAILABLE, f"MLX not available: {IMPORT_ERROR if not MLX_AVAILABLE else ''}")
class TestDeepseekOCRModel(unittest.TestCase):
    """Test DeepseekOCRForCausalLM model."""

    def setUp(self):
        """Set up test model."""
        self.config = DeepseekOCRConfig(
            image_token_id=128256,
            n_embed=128,  # Small for testing
            vocab_size=1000,
            hidden_size=128,
        )
        self.model = build_deepseek_ocr_model(self.config)

    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model.sam_model)
        self.assertIsNotNone(self.model.vision_model)
        self.assertIsNotNone(self.model.projector)
        self.assertIsNotNone(self.model.language_model)

    def test_special_tokens(self):
        """Test special token initialization."""
        self.assertEqual(self.model.image_newline.shape, (128,))
        self.assertEqual(self.model.view_separator.shape, (128,))

    def test_get_input_embeddings_text_only(self):
        """Test getting embeddings for text-only input."""
        input_ids = mx.array([[1, 50, 100, 200, 2]])  # [BOS, tokens..., EOS]

        embeds = self.model.get_input_embeddings(input_ids, vision_embeddings=None)

        # Should return text embeddings only
        self.assertEqual(embeds.shape[0], 1)  # Batch size
        self.assertEqual(embeds.shape[1], 5)  # Sequence length
        self.assertEqual(embeds.shape[2], 128)  # Hidden size

    def test_get_input_embeddings_with_vision(self):
        """Test merging vision embeddings into text."""
        # Input: [BOS, <image>, token, EOS]
        input_ids = mx.array([[1, 128256, 50, 2]])

        # Vision embeddings: 10 tokens
        vision_embeds = [mx.zeros((10, 128))]

        embeds = self.model.get_input_embeddings(input_ids, vision_embeds)

        # Total length: 1 (BOS) + 10 (vision) + 1 (token) + 1 (EOS) = 13
        self.assertEqual(embeds.shape[0], 1)
        self.assertEqual(embeds.shape[1], 13)
        self.assertEqual(embeds.shape[2], 128)

    def test_process_vision_input_global_only(self):
        """Test vision processing (global view only)."""
        # Create dummy vision input
        pixel_values = mx.random.normal((1, 3, 1280, 1280))
        images_crop = mx.zeros((1, 1, 3, 1024, 1024))  # No crops
        images_spatial_crop = mx.array([[1, 1]])  # 1x1 grid

        vision_embeds = self.model.process_vision_input(
            pixel_values, images_crop, images_spatial_crop
        )

        # Should return list with 1 embedding
        self.assertEqual(len(vision_embeds), 1)

        # Embedding should be 2D: [num_tokens, n_embed]
        self.assertEqual(vision_embeds[0].ndim, 2)
        self.assertEqual(vision_embeds[0].shape[1], 128)

    def test_forward_pass(self):
        """Test forward pass through model."""
        # Text-only forward pass
        input_ids = mx.array([[1, 50, 100, 2]])
        inputs_embeds = self.model.get_input_embeddings(input_ids)

        logits = self.model(inputs_embeds=inputs_embeds)

        # Should return logits
        self.assertEqual(logits.shape[0], 1)  # Batch size
        self.assertEqual(logits.shape[2], 1000)  # Vocab size


@unittest.skipUnless(MLX_AVAILABLE, f"MLX not available: {IMPORT_ERROR if not MLX_AVAILABLE else ''}")
class TestSampling(unittest.TestCase):
    """Test sampling strategies."""

    def test_temperature_greedy(self):
        """Test greedy sampling (temperature=0)."""
        logits = mx.array([[1.0, 5.0, 2.0, 3.0]])

        scaled = apply_temperature(logits, temperature=0.0)

        # Max index should have very high logit
        max_idx = mx.argmax(scaled, axis=-1)
        self.assertEqual(int(max_idx[0]), 1)  # Index 1 has highest logit

    def test_temperature_scaling(self):
        """Test temperature scaling."""
        logits = mx.array([[1.0, 2.0, 3.0]])

        # Higher temperature should make distribution more uniform
        scaled_low = apply_temperature(logits, temperature=0.5)
        scaled_high = apply_temperature(logits, temperature=2.0)

        # Check that scaling works
        self.assertEqual(scaled_low.shape, logits.shape)
        self.assertEqual(scaled_high.shape, logits.shape)

    def test_sample_token_greedy(self):
        """Test greedy token sampling."""
        logits = mx.array([[1.0, 5.0, 2.0]])
        config = SamplingConfig(temperature=0.0)

        token = sample_token(logits, config)

        # Should always select index 1 (highest logit)
        self.assertEqual(int(token[0]), 1)

    def test_sample_token_sampling(self):
        """Test stochastic token sampling."""
        logits = mx.array([[1.0, 2.0, 3.0]])
        config = SamplingConfig(temperature=1.0, top_p=1.0)

        token = sample_token(logits, config)

        # Should return a valid token index
        self.assertGreaterEqual(int(token[0]), 0)
        self.assertLess(int(token[0]), 3)


@unittest.skipUnless(MLX_AVAILABLE, f"MLX not available: {IMPORT_ERROR if not MLX_AVAILABLE else ''}")
class TestPipeline(unittest.TestCase):
    """Test complete inference pipeline."""

    def setUp(self):
        """Set up test pipeline."""
        config = DeepseekOCRConfig(
            image_token_id=128256,
            n_embed=128,
            vocab_size=1000,
            hidden_size=128,
        )
        model = build_deepseek_ocr_model(config)
        tokenizer = DummyTokenizer()
        processor = DeepseekOCRProcessor(
            tokenizer,
            image_size=1024,
            base_size=1280,
        )

        self.pipeline = DeepSeekOCRPipeline(model, processor, tokenizer)

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline.model)
        self.assertIsNotNone(self.pipeline.processor)
        self.assertIsNotNone(self.pipeline.tokenizer)

    def test_preprocess(self):
        """Test preprocessing through pipeline."""
        img = Image.new('RGB', (640, 640), color=(128, 128, 128))
        prompt = "Transcribe: <image>"

        inputs = self.pipeline.preprocess([img], prompt, cropping=False)

        self.assertIn('input_ids', inputs)
        self.assertIn('pixel_values', inputs)
        self.assertIn('images_crop', inputs)
        self.assertIn('images_spatial_crop', inputs)

    def test_forward_through_pipeline(self):
        """Test forward pass through pipeline."""
        img = Image.new('RGB', (640, 640), color=(128, 128, 128))
        prompt = "Transcribe: <image>"

        inputs = self.pipeline.preprocess([img], prompt, cropping=False)

        # Forward pass
        output = self.pipeline.forward(**inputs)

        # Should return logits
        self.assertIsNotNone(output)
        self.assertEqual(output.ndim, 3)  # [batch, seq, vocab]


@unittest.skipUnless(MLX_AVAILABLE, f"MLX not available: {IMPORT_ERROR if not MLX_AVAILABLE else ''}")
class TestIntegration(unittest.TestCase):
    """Integration tests for complete pipeline."""

    def test_end_to_end_inference(self):
        """Test complete end-to-end inference."""
        # Create small model
        config = DeepseekOCRConfig(
            image_token_id=128256,
            n_embed=128,
            vocab_size=1000,
            hidden_size=128,
        )
        model = build_deepseek_ocr_model(config)
        tokenizer = DummyTokenizer()
        processor = DeepseekOCRProcessor(tokenizer, image_size=1024, base_size=1280)
        pipeline = DeepSeekOCRPipeline(model, processor, tokenizer)

        # Create test image
        img = Image.new('RGB', (640, 640), color=(128, 128, 128))
        prompt = "OCR: <image>"

        # Run generation (just 1 token for speed)
        result = pipeline.generate(
            images=[img],
            prompt=prompt,
            max_tokens=1,
            temperature=0.0,
        )

        # Check result
        self.assertIn('text', result)
        self.assertIn('token_ids', result)
        self.assertIn('num_tokens', result)

    def test_multi_image_inference(self):
        """Test inference with multiple images."""
        config = DeepseekOCRConfig(
            image_token_id=128256,
            n_embed=128,
            vocab_size=1000,
            hidden_size=128,
        )
        model = build_deepseek_ocr_model(config)
        tokenizer = DummyTokenizer()
        processor = DeepseekOCRProcessor(tokenizer, image_size=1024, base_size=1280)
        pipeline = DeepSeekOCRPipeline(model, processor, tokenizer)

        # Create test images
        img1 = Image.new('RGB', (640, 640), color=(255, 0, 0))
        img2 = Image.new('RGB', (640, 640), color=(0, 255, 0))
        prompt = "Compare <image> and <image>"

        # Preprocess
        inputs = pipeline.preprocess([img1, img2], prompt, cropping=False)

        # Check that both images are processed
        self.assertEqual(inputs['pixel_values'].shape[0], 2)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_tests()
