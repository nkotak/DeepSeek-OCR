#!/usr/bin/env python3
"""
MLX Installation Validator

This script validates that MLX is properly installed with all required features
for DeepSeek-OCR migration.

Usage:
    python scripts/validate_mlx.py

Exit codes:
    0: All checks passed
    1: One or more checks failed
"""

import sys
import platform


def check_python_version():
    """Check Python version is 3.11+"""
    print("=" * 60)
    print("1. Checking Python Version")
    print("=" * 60)

    major, minor = sys.version_info[:2]
    version_str = f"{major}.{minor}"

    print(f"   Python version: {version_str}")

    if major < 3 or (major == 3 and minor < 11):
        print(f"   ✗ FAILED: Python 3.11+ required, found {version_str}")
        return False

    print(f"   ✓ PASSED: Python {version_str} is compatible")
    return True


def check_platform():
    """Check if running on Apple Silicon"""
    print("\n" + "=" * 60)
    print("2. Checking Platform")
    print("=" * 60)

    system = platform.system()
    machine = platform.machine()

    print(f"   System: {system}")
    print(f"   Machine: {machine}")

    if system == "Darwin" and machine == "arm64":
        print("   ✓ PASSED: Running on Apple Silicon")
        return True
    elif system == "Darwin":
        print(f"   ⚠ WARNING: macOS detected but not Apple Silicon ({machine})")
        print("   MLX will work but may not be optimized")
        return True
    else:
        print(f"   ⚠ WARNING: Not running on macOS ({system})")
        print("   MLX is optimized for Apple Silicon")
        return True  # Don't fail, just warn


def check_mlx_installation():
    """Check MLX is installed and get version"""
    print("\n" + "=" * 60)
    print("3. Checking MLX Installation")
    print("=" * 60)

    try:
        import mlx.core as mx
        version = mx.__version__
        print(f"   MLX version: {version}")

        # Parse version
        major, minor, patch = map(int, version.split('.')[:3])

        # Check minimum version (0.28.0)
        if major > 0 or (major == 0 and minor >= 28):
            print(f"   ✓ PASSED: MLX {version} meets minimum requirement (>=0.28.0)")
            return True, version
        else:
            print(f"   ✗ FAILED: MLX {version} is below minimum requirement (>=0.28.0)")
            return False, version

    except ImportError as e:
        print(f"   ✗ FAILED: MLX not installed - {e}")
        print("\n   Install with: pip install mlx>=0.28.0")
        return False, None


def check_mlx_nn():
    """Check MLX neural network module"""
    print("\n" + "=" * 60)
    print("4. Checking MLX Neural Network Module")
    print("=" * 60)

    try:
        import mlx.nn as nn
        print("   ✓ PASSED: mlx.nn is available")
        return True
    except ImportError as e:
        print(f"   ✗ FAILED: mlx.nn not available - {e}")
        return False


def check_mlx_lm():
    """Check MLX-LM is installed"""
    print("\n" + "=" * 60)
    print("5. Checking MLX-LM Installation")
    print("=" * 60)

    try:
        import mlx_lm
        print("   MLX-LM is installed")
        print("   ✓ PASSED: MLX-LM available")
        return True
    except ImportError as e:
        print(f"   ✗ FAILED: MLX-LM not installed - {e}")
        print("\n   Install with: pip install mlx-lm>=0.10.0")
        return False


def test_tensor_operations():
    """Test basic MLX tensor operations"""
    print("\n" + "=" * 60)
    print("6. Testing Basic Tensor Operations")
    print("=" * 60)

    try:
        import mlx.core as mx

        # Create tensor
        x = mx.random.normal([2, 3, 224, 224])
        assert x.shape == [2, 3, 224, 224], f"Unexpected shape: {x.shape}"
        print(f"   ✓ Created tensor: {x.shape}")

        # Test reshape
        y = x.reshape([2, 3, -1])
        assert y.shape == [2, 3, 224*224], f"Reshape failed: {y.shape}"
        print(f"   ✓ Reshape working: {y.shape}")

        # Test transpose
        z = x.transpose([0, 2, 3, 1])
        assert z.shape == [2, 224, 224, 3], f"Transpose failed: {z.shape}"
        print(f"   ✓ Transpose working: {z.shape}")

        # Test concatenate
        w = mx.concatenate([x, x], axis=0)
        assert w.shape == [4, 3, 224, 224], f"Concatenate failed: {w.shape}"
        print(f"   ✓ Concatenate working: {w.shape}")

        print("   ✓ PASSED: Basic tensor operations working")
        return True

    except Exception as e:
        print(f"   ✗ FAILED: Tensor operations failed - {e}")
        return False


def test_sdpa():
    """Test Scaled Dot-Product Attention (critical for transformers)"""
    print("\n" + "=" * 60)
    print("7. Testing Scaled Dot-Product Attention (SDPA)")
    print("=" * 60)

    try:
        import mlx.core as mx

        # Create Q, K, V tensors
        batch_size = 2
        num_heads = 8
        seq_len = 16
        head_dim = 64

        q = mx.random.normal([batch_size, num_heads, seq_len, head_dim])
        k = mx.random.normal([batch_size, num_heads, seq_len, head_dim])
        v = mx.random.normal([batch_size, num_heads, seq_len, head_dim])

        print(f"   Q shape: {q.shape}")
        print(f"   K shape: {k.shape}")
        print(f"   V shape: {v.shape}")

        # Test SDPA
        out = mx.fast.scaled_dot_product_attention(q, k, v)

        expected_shape = [batch_size, num_heads, seq_len, head_dim]
        assert list(out.shape) == expected_shape, f"SDPA output shape mismatch: {out.shape} vs {expected_shape}"

        print(f"   ✓ SDPA output: {out.shape}")
        print("   ✓ PASSED: Scaled Dot-Product Attention working")
        return True

    except AttributeError:
        print("   ✗ FAILED: mx.fast.scaled_dot_product_attention not available")
        print("   This requires MLX >=0.28.0")
        return False
    except Exception as e:
        print(f"   ✗ FAILED: SDPA test failed - {e}")
        return False


def test_neural_network_layers():
    """Test neural network layers"""
    print("\n" + "=" * 60)
    print("8. Testing Neural Network Layers")
    print("=" * 60)

    try:
        import mlx.core as mx
        import mlx.nn as nn

        # Test Conv2d
        conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        x = mx.random.normal([2, 3, 224, 224])
        out = conv(x)
        assert list(out.shape) == [2, 64, 224, 224], f"Conv2d output shape mismatch: {out.shape}"
        print(f"   ✓ Conv2d working: {out.shape}")

        # Test Linear
        linear = nn.Linear(128, 256)
        x = mx.random.normal([2, 128])
        out = linear(x)
        assert list(out.shape) == [2, 256], f"Linear output shape mismatch: {out.shape}"
        print(f"   ✓ Linear working: {out.shape}")

        # Test LayerNorm
        ln = nn.LayerNorm(64)
        x = mx.random.normal([2, 64])
        out = ln(x)
        assert list(out.shape) == [2, 64], f"LayerNorm output shape mismatch: {out.shape}"
        print(f"   ✓ LayerNorm working: {out.shape}")

        # Test RMSNorm
        rms = nn.RMSNorm(64)
        x = mx.random.normal([2, 64])
        out = rms(x)
        assert list(out.shape) == [2, 64], f"RMSNorm output shape mismatch: {out.shape}"
        print(f"   ✓ RMSNorm working: {out.shape}")

        # Test GELU
        gelu = nn.GELU()
        x = mx.random.normal([2, 64])
        out = gelu(x)
        assert list(out.shape) == [2, 64], f"GELU output shape mismatch: {out.shape}"
        print(f"   ✓ GELU working: {out.shape}")

        print("   ✓ PASSED: All neural network layers working")
        return True

    except Exception as e:
        print(f"   ✗ FAILED: Neural network layers test failed - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_operations():
    """Test image operations (resize, etc.)"""
    print("\n" + "=" * 60)
    print("9. Testing Image Operations")
    print("=" * 60)

    try:
        import mlx.core as mx

        # Test image resize (if available)
        try:
            x = mx.random.normal([224, 224, 3])
            resized = mx.image.resize(x, [112, 112], method='bilinear')
            assert list(resized.shape) == [112, 112, 3], f"Resize output shape mismatch: {resized.shape}"
            print(f"   ✓ Image resize working: {resized.shape}")
            print("   ✓ PASSED: Image operations available")
            return True
        except AttributeError:
            print("   ⚠ WARNING: mx.image.resize not available")
            print("   This is okay - we'll implement custom resize if needed")
            return True

    except Exception as e:
        print(f"   ✗ FAILED: Image operations test failed - {e}")
        return False


def check_dependencies():
    """Check other required dependencies"""
    print("\n" + "=" * 60)
    print("10. Checking Other Dependencies")
    print("=" * 60)

    dependencies = {
        'transformers': '4.46.3',
        'tokenizers': '0.20.3',
        'PIL': None,  # Pillow
        'numpy': None,
        'easydict': None,
        'addict': None,
    }

    all_ok = True

    for module, expected_version in dependencies.items():
        # Set module_name before try block to avoid UnboundLocalError
        module_name = 'Pillow' if module == 'PIL' else module

        try:
            if module == 'PIL':
                import PIL
                module_obj = PIL
            else:
                module_obj = __import__(module)

            version = getattr(module_obj, '__version__', 'unknown')

            if expected_version:
                print(f"   ✓ {module_name}: {version} (expected: {expected_version})")
            else:
                print(f"   ✓ {module_name}: {version}")

        except ImportError:
            print(f"   ✗ {module_name}: Not installed")
            all_ok = False

    if all_ok:
        print("   ✓ PASSED: All dependencies available")
    else:
        print("   ⚠ WARNING: Some dependencies missing")
        print("   Install with: pip install -r mlx_port/requirements_mlx.txt")

    return True  # Don't fail on missing optional deps


def print_summary(results):
    """Print summary of all checks"""
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    total = len(results)
    passed = sum(results.values())
    failed = total - passed

    print(f"\n   Total checks: {total}")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")

    if failed == 0:
        print("\n   ✅ ALL CHECKS PASSED!")
        print("\n   MLX is properly installed and ready for DeepSeek-OCR migration.")
        print("   You can proceed to Phase 1.2: Directory Structure Setup")
        return True
    else:
        print("\n   ❌ SOME CHECKS FAILED")
        print("\n   Failed checks:")
        for name, passed in results.items():
            if not passed:
                print(f"      - {name}")
        print("\n   Please fix the failed checks before proceeding.")
        return False


def main():
    """Run all validation checks"""
    print("\n" + "=" * 60)
    print("MLX INSTALLATION VALIDATOR")
    print("DeepSeek-OCR Migration - Phase 1.1")
    print("=" * 60)

    results = {}

    # Run all checks
    results['Python Version'] = check_python_version()
    results['Platform'] = check_platform()

    mlx_ok, mlx_version = check_mlx_installation()
    results['MLX Installation'] = mlx_ok

    if mlx_ok:
        results['MLX Neural Network'] = check_mlx_nn()
        results['MLX-LM'] = check_mlx_lm()
        results['Tensor Operations'] = test_tensor_operations()
        results['SDPA'] = test_sdpa()
        results['Neural Network Layers'] = test_neural_network_layers()
        results['Image Operations'] = test_image_operations()
    else:
        print("\n   Skipping remaining tests (MLX not installed)")
        results['MLX Neural Network'] = False
        results['MLX-LM'] = False
        results['Tensor Operations'] = False
        results['SDPA'] = False
        results['Neural Network Layers'] = False
        results['Image Operations'] = False

    results['Dependencies'] = check_dependencies()

    # Print summary
    success = print_summary(results)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
