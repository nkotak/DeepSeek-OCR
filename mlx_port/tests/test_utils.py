"""
Testing utilities for DeepSeek-OCR MLX port

This module provides utilities for comparing PyTorch and MLX outputs,
loading weights, and other testing helpers.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path


class PyTorchMLXComparator:
    """
    Utility class for comparing PyTorch and MLX outputs

    This class provides methods to:
    - Convert between PyTorch tensors and MLX arrays
    - Compare outputs with configurable tolerances
    - Load PyTorch weights into MLX modules
    - Compare layer outputs
    """

    @staticmethod
    def torch_to_mlx(torch_tensor):
        """
        Convert PyTorch tensor to MLX array

        Args:
            torch_tensor: PyTorch tensor (torch.Tensor)

        Returns:
            MLX array (mx.array)
        """
        import mlx.core as mx
        return mx.array(torch_tensor.detach().cpu().numpy())

    @staticmethod
    def mlx_to_torch(mlx_array):
        """
        Convert MLX array to PyTorch tensor

        Args:
            mlx_array: MLX array (mx.array)

        Returns:
            PyTorch tensor (torch.Tensor)
        """
        import torch
        return torch.from_numpy(np.array(mlx_array))

    @staticmethod
    def torch_to_numpy(torch_tensor):
        """
        Convert PyTorch tensor to NumPy array

        Args:
            torch_tensor: PyTorch tensor

        Returns:
            NumPy array
        """
        return torch_tensor.detach().cpu().numpy()

    @staticmethod
    def mlx_to_numpy(mlx_array):
        """
        Convert MLX array to NumPy array

        Args:
            mlx_array: MLX array

        Returns:
            NumPy array
        """
        return np.array(mlx_array)

    @staticmethod
    def assert_close(
        torch_out,
        mlx_out,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        name: str = "",
        verbose: bool = True
    ):
        """
        Assert that PyTorch and MLX outputs are close

        Args:
            torch_out: PyTorch output tensor
            mlx_out: MLX output array
            rtol: Relative tolerance
            atol: Absolute tolerance
            name: Name for logging (optional)
            verbose: Whether to print detailed comparison info

        Raises:
            AssertionError: If outputs differ beyond tolerance
        """
        # Convert to numpy
        torch_np = PyTorchMLXComparator.torch_to_numpy(torch_out)
        mlx_np = PyTorchMLXComparator.mlx_to_numpy(mlx_out)

        # Check shapes
        if torch_np.shape != mlx_np.shape:
            raise AssertionError(
                f"{name}: Shape mismatch - "
                f"PyTorch: {torch_np.shape}, MLX: {mlx_np.shape}"
            )

        # Compute differences
        abs_diff = np.abs(torch_np - mlx_np)
        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)

        # Compute relative differences
        torch_abs = np.abs(torch_np)
        rel_diff = np.abs((torch_np - mlx_np) / (torch_abs + 1e-10))
        max_rel_diff = np.max(rel_diff)
        mean_rel_diff = np.mean(rel_diff)

        # Find maximum difference locations
        max_diff_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)

        if verbose:
            print(f"\n{'='*70}")
            print(f"Comparison: {name or 'Unnamed'}")
            print(f"{'='*70}")
            print(f"  Shape: {torch_np.shape}")
            print(f"  Dtype: PyTorch={torch_out.dtype}, MLX={mlx_out.dtype}")
            print(f"\n  Absolute Differences:")
            print(f"    Max:  {max_abs_diff:.6e} (at index {max_diff_idx})")
            print(f"    Mean: {mean_abs_diff:.6e}")
            print(f"\n  Relative Differences:")
            print(f"    Max:  {max_rel_diff:.6e}")
            print(f"    Mean: {mean_rel_diff:.6e}")
            print(f"\n  Tolerance:")
            print(f"    rtol: {rtol:.6e}")
            print(f"    atol: {atol:.6e}")
            print(f"\n  Sample values at max diff location:")
            print(f"    PyTorch: {torch_np[max_diff_idx]:.6e}")
            print(f"    MLX:     {mlx_np[max_diff_idx]:.6e}")

        # Assert closeness
        try:
            np.testing.assert_allclose(
                torch_np, mlx_np,
                rtol=rtol, atol=atol,
                err_msg=f"{name}: Outputs differ beyond tolerance"
            )
            if verbose:
                print(f"\n  ✅ PASSED (within tolerance)")
                print(f"{'='*70}\n")
            return True
        except AssertionError as e:
            if verbose:
                print(f"\n  ❌ FAILED (exceeds tolerance)")
                print(f"{'='*70}\n")
            raise e

    @staticmethod
    def load_pytorch_weights_to_mlx(
        mlx_module,
        pytorch_state_dict: Dict[str, Any],
        prefix: str = "",
        strict: bool = True,
        verbose: bool = True
    ):
        """
        Load PyTorch weights into MLX module

        Args:
            mlx_module: MLX module to load weights into
            pytorch_state_dict: PyTorch state dict
            prefix: Prefix to remove from keys
            strict: Whether to raise error on missing/unexpected keys
            verbose: Whether to print loading info

        Returns:
            MLX module with loaded weights

        Raises:
            ValueError: If strict=True and there are missing keys
        """
        import mlx.core as mx

        # Get MLX module parameters
        mlx_params = dict(mlx_module.parameters())
        loaded_keys = set()
        missing_keys = set()
        unexpected_keys = set()

        if verbose:
            print(f"\n{'='*70}")
            print("Loading PyTorch weights into MLX module")
            print(f"{'='*70}")
            print(f"  PyTorch state dict keys: {len(pytorch_state_dict)}")
            print(f"  MLX parameter keys: {len(mlx_params)}")
            if prefix:
                print(f"  Removing prefix: '{prefix}'")

        # Load weights
        for key, value in pytorch_state_dict.items():
            # Remove prefix if specified
            mlx_key = key
            if prefix and key.startswith(prefix):
                mlx_key = key[len(prefix):]

            if mlx_key in mlx_params:
                # Convert PyTorch tensor to MLX array
                if hasattr(value, 'detach'):
                    # It's a PyTorch tensor
                    np_value = value.detach().cpu().numpy()
                else:
                    # Already numpy or other format
                    np_value = np.array(value)

                mlx_params[mlx_key] = mx.array(np_value)
                loaded_keys.add(mlx_key)
            else:
                unexpected_keys.add(key)

        # Check for missing keys
        for key in mlx_params:
            if key not in loaded_keys:
                missing_keys.add(key)

        if verbose:
            print(f"\n  Loaded keys: {len(loaded_keys)}")

            if unexpected_keys:
                print(f"  Unexpected keys: {len(unexpected_keys)}")
                for key in list(unexpected_keys)[:5]:
                    print(f"    - {key}")
                if len(unexpected_keys) > 5:
                    print(f"    ... and {len(unexpected_keys) - 5} more")

            if missing_keys:
                print(f"  Missing keys: {len(missing_keys)}")
                for key in list(missing_keys)[:5]:
                    print(f"    - {key}")
                if len(missing_keys) > 5:
                    print(f"    ... and {len(missing_keys) - 5} more")

        if strict and missing_keys:
            raise ValueError(
                f"Missing keys in MLX module: {', '.join(list(missing_keys)[:10])}"
                + (f" ... and {len(missing_keys) - 10} more" if len(missing_keys) > 10 else "")
            )

        # Update module parameters
        mlx_module.update(mlx_params)

        if verbose:
            print(f"\n  ✅ Weights loaded successfully")
            print(f"{'='*70}\n")

        return mlx_module

    @staticmethod
    def compare_layer_outputs(
        torch_layer,
        mlx_layer,
        input_tensor,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        name: str = "Layer"
    ) -> Tuple:
        """
        Compare outputs of PyTorch and MLX layers

        Args:
            torch_layer: PyTorch layer
            mlx_layer: MLX layer
            input_tensor: Input tensor (PyTorch)
            rtol: Relative tolerance
            atol: Absolute tolerance
            name: Layer name for logging

        Returns:
            Tuple of (PyTorch output, MLX output)
        """
        import torch

        # Convert input
        mlx_input = PyTorchMLXComparator.torch_to_mlx(input_tensor)

        # Forward pass
        with torch.no_grad():
            torch_output = torch_layer(input_tensor)

        mlx_output = mlx_layer(mlx_input)

        # Compare
        PyTorchMLXComparator.assert_close(
            torch_output, mlx_output,
            rtol=rtol, atol=atol,
            name=name
        )

        return torch_output, mlx_output

    @staticmethod
    def compute_statistics(tensor: Union[np.ndarray, Any]) -> Dict[str, float]:
        """
        Compute statistics for a tensor

        Args:
            tensor: Tensor (NumPy, PyTorch, or MLX)

        Returns:
            Dictionary of statistics
        """
        # Convert to numpy
        if hasattr(tensor, 'detach'):
            # PyTorch
            arr = tensor.detach().cpu().numpy()
        elif hasattr(tensor, '__array__'):
            # MLX or NumPy
            arr = np.array(tensor)
        else:
            arr = tensor

        return {
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'median': float(np.median(arr)),
            'shape': arr.shape,
            'dtype': str(arr.dtype),
        }


class TestDataGenerator:
    """Generate test data for various scenarios"""

    @staticmethod
    def create_random_image(
        shape: Tuple[int, ...],
        seed: Optional[int] = None,
        framework: str = 'mlx'
    ):
        """
        Create random image tensor

        Args:
            shape: Image shape (B, C, H, W)
            seed: Random seed
            framework: 'mlx' or 'torch'

        Returns:
            Random image tensor
        """
        if framework == 'mlx':
            import mlx.core as mx
            if seed is not None:
                mx.random.seed(seed)
            return mx.random.normal(shape)
        elif framework == 'torch':
            import torch
            if seed is not None:
                torch.manual_seed(seed)
            return torch.randn(shape)
        else:
            raise ValueError(f"Unknown framework: {framework}")

    @staticmethod
    def create_paired_random_images(
        shape: Tuple[int, ...],
        seed: Optional[int] = None
    ) -> Tuple:
        """
        Create random image tensors in both PyTorch and MLX (same data)

        Args:
            shape: Image shape (B, C, H, W)
            seed: Random seed

        Returns:
            Tuple of (torch_tensor, mlx_array)
        """
        import torch
        import mlx.core as mx

        # Generate with PyTorch
        if seed is not None:
            torch.manual_seed(seed)
        torch_tensor = torch.randn(shape)

        # Convert to MLX
        mlx_array = mx.array(torch_tensor.numpy())

        return torch_tensor, mlx_array


class BenchmarkHelper:
    """Helper for benchmarking MLX vs PyTorch"""

    @staticmethod
    def time_forward_pass(
        model,
        input_data,
        num_iterations: int = 10,
        warmup: int = 3,
        framework: str = 'mlx'
    ) -> Dict[str, float]:
        """
        Time forward pass of a model

        Args:
            model: Model to benchmark
            input_data: Input data
            num_iterations: Number of timing iterations
            warmup: Number of warmup iterations
            framework: 'mlx' or 'torch'

        Returns:
            Dictionary with timing statistics
        """
        import time

        if framework == 'torch':
            import torch
            device = next(model.parameters()).device
            input_data = input_data.to(device)

        # Warmup
        for _ in range(warmup):
            if framework == 'torch':
                with torch.no_grad():
                    _ = model(input_data)
            else:
                _ = model(input_data)

        # Synchronize before timing
        if framework == 'torch':
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        else:
            import mlx.core as mx
            mx.eval(input_data)

        # Time iterations
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()

            if framework == 'torch':
                with torch.no_grad():
                    output = model(input_data)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            else:
                import mlx.core as mx
                output = model(input_data)
                mx.eval(output)

            end = time.perf_counter()
            times.append(end - start)

        times = np.array(times)

        return {
            'mean': float(np.mean(times)),
            'std': float(np.std(times)),
            'min': float(np.min(times)),
            'max': float(np.max(times)),
            'median': float(np.median(times)),
            'times': times.tolist(),
        }


def assert_shapes_equal(shape1, shape2, name: str = ""):
    """
    Assert two shapes are equal

    Args:
        shape1: First shape
        shape2: Second shape
        name: Name for error message
    """
    assert list(shape1) == list(shape2), \
        f"{name} Shape mismatch: {shape1} vs {shape2}"


def print_tensor_info(tensor, name: str = "Tensor"):
    """
    Print information about a tensor

    Args:
        tensor: Tensor to inspect
        name: Name for display
    """
    stats = PyTorchMLXComparator.compute_statistics(tensor)

    print(f"\n{name}:")
    print(f"  Shape: {stats['shape']}")
    print(f"  Dtype: {stats['dtype']}")
    print(f"  Min: {stats['min']:.6e}")
    print(f"  Max: {stats['max']:.6e}")
    print(f"  Mean: {stats['mean']:.6e}")
    print(f"  Std: {stats['std']:.6e}")
    print(f"  Median: {stats['median']:.6e}")


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'PyTorchMLXComparator',
    'TestDataGenerator',
    'BenchmarkHelper',
    'assert_shapes_equal',
    'print_tensor_info',
]
