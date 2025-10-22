"""
pytest configuration for DeepSeek-OCR MLX tests

This module configures pytest and provides fixtures for all test modules.
"""

import pytest
import sys
import os
from pathlib import Path
import warnings

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers",
        "validation: marks tests as validation tests (require PyTorch)"
    )
    config.addinivalue_line(
        "markers",
        "requires_pytorch: marks tests that require PyTorch"
    )
    config.addinivalue_line(
        "markers",
        "requires_model: marks tests that require model weights"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip tests based on availability"""

    # Check if PyTorch is available
    try:
        import torch
        has_pytorch = True
    except ImportError:
        has_pytorch = False

    # Skip markers
    skip_pytorch = pytest.mark.skip(reason="PyTorch not available")

    for item in items:
        # Skip PyTorch-dependent tests if PyTorch not available
        if "requires_pytorch" in item.keywords and not has_pytorch:
            item.add_marker(skip_pytorch)
        if "validation" in item.keywords and not has_pytorch:
            item.add_marker(skip_pytorch)


# ============================================================================
# Session-scope Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def mlx_available():
    """Check if MLX is available"""
    try:
        import mlx.core as mx
        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def pytorch_available():
    """Check if PyTorch is available"""
    try:
        import torch
        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def test_data_dir():
    """Directory containing test data"""
    data_dir = Path(__file__).parent / "fixtures"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture(scope="session")
def output_dir(tmp_path_factory):
    """Temporary directory for test outputs"""
    return tmp_path_factory.mktemp("test_outputs")


@pytest.fixture(scope="session")
def model_cache_dir(tmp_path_factory):
    """Temporary directory for model cache"""
    return tmp_path_factory.mktemp("model_cache")


# ============================================================================
# Module-scope Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def pytorch_model_path():
    """Path to PyTorch model (HuggingFace identifier)"""
    return "deepseek-ai/DeepSeek-OCR"


# ============================================================================
# Function-scope Fixtures (Common Test Data)
# ============================================================================

@pytest.fixture
def random_seed():
    """Random seed for reproducible tests"""
    return 42


@pytest.fixture
def small_image_shape():
    """Shape for small test images"""
    return (2, 3, 224, 224)  # [B, C, H, W]


@pytest.fixture
def large_image_shape():
    """Shape for large test images (DeepSeek-OCR size)"""
    return (2, 3, 1024, 1024)  # [B, C, H, W]


@pytest.fixture
def tolerance():
    """Default tolerance for numerical comparisons"""
    from mlx_port.config_mlx import TEST_TOLERANCE_RTOL, TEST_TOLERANCE_ATOL
    return {
        "rtol": TEST_TOLERANCE_RTOL,
        "atol": TEST_TOLERANCE_ATOL
    }


@pytest.fixture
def relaxed_tolerance():
    """Relaxed tolerance for interpolation/resize operations"""
    return {
        "rtol": 1e-3,
        "atol": 1e-4
    }


# ============================================================================
# MLX-specific Fixtures
# ============================================================================

@pytest.fixture
def mlx_random_image_small(small_image_shape, random_seed):
    """Random MLX image tensor (small)"""
    import mlx.core as mx
    mx.random.seed(random_seed)
    return mx.random.normal(small_image_shape)


@pytest.fixture
def mlx_random_image_large(large_image_shape, random_seed):
    """Random MLX image tensor (large)"""
    import mlx.core as mx
    mx.random.seed(random_seed)
    return mx.random.normal(large_image_shape)


# ============================================================================
# PyTorch-specific Fixtures
# ============================================================================

@pytest.fixture
def pytorch_random_image_small(small_image_shape, random_seed):
    """Random PyTorch image tensor (small)"""
    try:
        import torch
        torch.manual_seed(random_seed)
        return torch.randn(small_image_shape)
    except ImportError:
        pytest.skip("PyTorch not available")


@pytest.fixture
def pytorch_random_image_large(large_image_shape, random_seed):
    """Random PyTorch image tensor (large)"""
    try:
        import torch
        torch.manual_seed(random_seed)
        return torch.randn(large_image_shape)
    except ImportError:
        pytest.skip("PyTorch not available")


# ============================================================================
# Paired Fixtures (PyTorch + MLX with same data)
# ============================================================================

@pytest.fixture
def paired_random_images_small(small_image_shape, random_seed):
    """
    Random image tensors in both PyTorch and MLX (same data)

    Returns:
        Tuple[torch.Tensor, mx.array]: (pytorch_tensor, mlx_array)
    """
    try:
        import torch
        import mlx.core as mx
        import numpy as np

        # Generate with PyTorch
        torch.manual_seed(random_seed)
        torch_tensor = torch.randn(small_image_shape)

        # Convert to MLX
        mlx_array = mx.array(torch_tensor.numpy())

        return torch_tensor, mlx_array

    except ImportError as e:
        pytest.skip(f"Required library not available: {e}")


@pytest.fixture
def paired_random_images_large(large_image_shape, random_seed):
    """
    Random image tensors in both PyTorch and MLX (same data)

    Returns:
        Tuple[torch.Tensor, mx.array]: (pytorch_tensor, mlx_array)
    """
    try:
        import torch
        import mlx.core as mx
        import numpy as np

        # Generate with PyTorch
        torch.manual_seed(random_seed)
        torch_tensor = torch.randn(large_image_shape)

        # Convert to MLX
        mlx_array = mx.array(torch_tensor.numpy())

        return torch_tensor, mlx_array

    except ImportError as e:
        pytest.skip(f"Required library not available: {e}")


# ============================================================================
# Test Image Fixtures
# ============================================================================

@pytest.fixture
def test_image_path(test_data_dir):
    """
    Path to a test image file (creates one if doesn't exist)

    Returns:
        Path: Path to test image
    """
    from PIL import Image
    import numpy as np

    image_path = test_data_dir / "test_image.jpg"

    if not image_path.exists():
        # Create a simple test image (256x256 RGB)
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(image_path)

    return image_path


@pytest.fixture
def test_image_pil(test_image_path):
    """
    Load test image as PIL Image

    Returns:
        PIL.Image: Test image
    """
    from PIL import Image
    return Image.open(test_image_path).convert('RGB')


# ============================================================================
# Benchmark Fixtures
# ============================================================================

@pytest.fixture
def benchmark_iterations():
    """Number of iterations for benchmarks"""
    return 10


@pytest.fixture
def benchmark_warmup():
    """Number of warmup iterations for benchmarks"""
    return 3


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_mlx_cache():
    """
    Clean up MLX cache after each test

    This prevents memory buildup during test runs
    """
    yield
    # Cleanup happens after test
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
    except Exception:
        pass  # Ignore errors during cleanup


# ============================================================================
# Utility Functions for Tests
# ============================================================================

def assert_shapes_equal(shape1, shape2, name=""):
    """Assert two shapes are equal"""
    assert list(shape1) == list(shape2), \
        f"{name} Shape mismatch: {shape1} vs {shape2}"


def skip_if_no_pytorch():
    """Skip test if PyTorch is not available"""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")


def skip_if_no_mlx():
    """Skip test if MLX is not available"""
    try:
        import mlx.core as mx
    except ImportError:
        pytest.skip("MLX not available")


def skip_if_no_model_weights():
    """Skip test if model weights are not available"""
    # Check if model weights are accessible
    import os
    if not os.environ.get('DEEPSEEK_OCR_MODEL_AVAILABLE'):
        pytest.skip("Model weights not available")


# Export utility functions
pytest.assert_shapes_equal = assert_shapes_equal
pytest.skip_if_no_pytorch = skip_if_no_pytorch
pytest.skip_if_no_mlx = skip_if_no_mlx
pytest.skip_if_no_model_weights = skip_if_no_model_weights
