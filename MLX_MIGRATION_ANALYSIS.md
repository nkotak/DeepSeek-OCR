# DeepSeek-OCR to MLX Migration Analysis

## Executive Summary

This document provides a comprehensive analysis of adapting DeepSeek-OCR from PyTorch/CUDA to MLX for Apple Silicon. Based on MLX releases 0.26.0 through 0.29.3, MLX now includes significant CUDA-equivalent operations that make this migration feasible.

## MLX CUDA Support Overview (v0.26.0 - v0.29.3)

### Version Highlights

#### v0.29.0 (August 2025)
- **MXFP4 quantization** support (Metal, CPU)
- Performance improvements in CUDA backend
- **mx.distributed** supports NCCL backend for CUDA
- Custom CUDA kernel support
- CUDA graph support
- Improved convolution operations
- Enhanced normalization layers (LayerNorm, RMSNorm)

#### v0.28.0 (August 2025)
- **First version of fused SDPA (Scaled Dot-Product Attention) for CUDA**
- **Convolutions in CUDA**
- Speed improvements in CUDA normalization layers
- Softmax optimizations
- Compiled kernel improvements

#### v0.27.1 (July 2025)
- **Initial PyPI release of CUDA backend**
- Works well for LLM inference
- Single-machine training and LoRA fine-tuning support
- Matmul, unary ops, binary ops, sort, random, copy ops
- Reduce, argreduce operations
- Softmax, LayerNorm, RMSNorm
- Compile support, indexing operations
- RoPE (Rotary Position Embedding)
- CUDA graphs for performance

#### v0.26.0 (June 2025)
- 5-bit quantization
- Initial CUDA backend backbone
- Complex operations support
- FFT operations
- Convolution operations

## DeepSeek-OCR Architecture Analysis

### Core Components

1. **Vision Encoders**
   - SAM ViT-B encoder (`sam_vary_sdpa.py`)
   - CLIP-L encoder (`clip_sdpa.py`)
   - Both use transformer architecture with attention mechanisms

2. **Projector**
   - MLP-based projection layer (`build_linear.py`)
   - Linear transformations

3. **Image Processing**
   - Dynamic image preprocessing
   - Cropping and tiling
   - Normalization

4. **Language Model Integration**
   - vLLM-based inference
   - DeepSeek V2/V3 language models

## Critical PyTorch/CUDA Operations Used

### 1. Tensor Operations
- `torch.Tensor` creation and manipulation
- `torch.zeros`, `torch.ones`, `torch.randn`
- `torch.cat`, `torch.stack`
- `.view()`, `.reshape()`, `.permute()`, `.transpose()`
- `.flatten()`, `.unsqueeze()`, `.squeeze()`
- Device management: `.cuda()`, `.to(device)`
- Dtype conversion: `.to(torch.bfloat16)`, `.to(torch.float32)`

### 2. Neural Network Operations
- `torch.nn.Linear` - Linear layers
- `torch.nn.Conv2d` - 2D convolutions
- `torch.nn.LayerNorm` - Layer normalization
- `torch.nn.Parameter` - Trainable parameters
- `torch.nn.GELU`, `torch.nn.ReLU` - Activation functions
- `torch.nn.Embedding` - Embedding layers

### 3. Functional Operations
- `F.interpolate` - Bilinear/bicubic interpolation for resizing
- `F.pad` - Padding operations
- `F.unfold` - Extract sliding local blocks
- `F.scaled_dot_product_attention` - Attention mechanism (PyTorch 2.0+)
- Various activation functions

### 4. Flash Attention
- `flash_attn_qkvpacked_func` - Packed QKV flash attention
- `flash_attn_func` - Standard flash attention
- Critical for performance in transformer blocks

### 5. Mathematical Operations
- `torch.einsum` - Einstein summation
- `torch.sigmoid`, `torch.sqrt`, `torch.exp`
- Matrix operations, element-wise ops

### 6. Image Processing
- Integration with PIL (Pillow)
- Torchvision transforms

## MLX Equivalents Mapping

### Basic Tensor Operations

| PyTorch Operation | MLX Equivalent | Status | Notes |
|------------------|----------------|--------|-------|
| `torch.Tensor` | `mx.array` | ✅ Available | Core MLX type |
| `torch.zeros()` | `mx.zeros()` | ✅ Available | |
| `torch.ones()` | `mx.ones()` | ✅ Available | |
| `torch.randn()` | `mx.random.normal()` | ✅ Available | |
| `.cuda()` | Automatic on Apple Silicon | ✅ Available | MLX handles device automatically |
| `.to(torch.bfloat16)` | `.astype(mx.bfloat16)` | ✅ Available | |
| `.view()` / `.reshape()` | `.reshape()` | ✅ Available | |
| `.permute()` | `.transpose()` | ✅ Available | |
| `.flatten()` | `.flatten()` | ✅ Available | |
| `torch.cat()` | `mx.concatenate()` | ✅ Available | |
| `torch.stack()` | `mx.stack()` | ✅ Available | |

### Neural Network Layers

| PyTorch Layer | MLX Equivalent | Status | Notes |
|--------------|----------------|--------|-------|
| `nn.Linear` | `nn.Linear` | ✅ Available | In mlx.nn |
| `nn.Conv2d` | `nn.Conv2d` | ✅ Available | CUDA support from v0.28.0 |
| `nn.LayerNorm` | `nn.LayerNorm` | ✅ Available | Optimized in v0.28.0+ |
| `nn.RMSNorm` | `nn.RMSNorm` | ✅ Available | |
| `nn.GELU` | `nn.GELU` | ✅ Available | |
| `nn.Parameter` | Standard array | ✅ Available | MLX handles trainability differently |
| `nn.Embedding` | `nn.Embedding` | ✅ Available | |

### Functional Operations

| PyTorch Function | MLX Equivalent | Status | Notes |
|-----------------|----------------|--------|-------|
| `F.interpolate` | `mx.image.resize` | ✅ Available | Bilinear, bicubic modes |
| `F.pad` | `mx.pad` | ✅ Available | Multiple padding modes |
| `F.unfold` | Custom implementation | ⚠️ Needs Implementation | Can be implemented with array ops |
| `F.scaled_dot_product_attention` | `mx.fast.scaled_dot_product_attention` | ✅ Available | Added in v0.28.0 |
| `F.softmax` | `mx.softmax` | ✅ Available | Optimized in CUDA backend |

### Attention Mechanisms

| PyTorch Operation | MLX Equivalent | Status | Notes |
|------------------|----------------|--------|-------|
| `flash_attn_qkvpacked_func` | `mx.fast.scaled_dot_product_attention` | ✅ Available | From v0.28.0 |
| Manual attention computation | `mx.fast.scaled_dot_product_attention` | ✅ Available | More efficient than manual |
| Relative position embeddings | Manual implementation | ⚠️ Needs Implementation | Can be implemented |

### Mathematical Operations

| PyTorch Operation | MLX Equivalent | Status | Notes |
|------------------|----------------|--------|-------|
| `torch.einsum` | `mx.einsum` | ✅ Available | |
| `torch.sigmoid` | `mx.sigmoid` | ✅ Available | |
| `torch.sqrt` | `mx.sqrt` | ✅ Available | |
| `torch.exp` | `mx.exp` | ✅ Available | |
| Matrix multiplication | `@` operator or `mx.matmul` | ✅ Available | Optimized in CUDA |

## Required Changes by File

### 1. `/DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepencoder/sam_vary_sdpa.py`

**Line-by-Line Changes:**

#### Imports (Lines 1-13)
```python
# CURRENT:
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_qkvpacked_func

# CHANGE TO:
import mlx.core as mx
import mlx.nn as nn
# Note: F.interpolate will need to use mx.image.resize
# flash_attn will use mx.fast.scaled_dot_product_attention
```

#### get_abs_pos function (Lines 19-38)
```python
# CURRENT (Lines 27-34):
old_pos_embed = old_pos_embed.to(torch.float32)
new_pos_embed = F.interpolate(
    old_pos_embed,
    size=(tgt_size, tgt_size),
    mode='bicubic',
    antialias=True,
    align_corners=False,
).to(dtype)

# CHANGE TO:
old_pos_embed = old_pos_embed.astype(mx.float32)
new_pos_embed = mx.image.resize(
    old_pos_embed,
    [tgt_size, tgt_size],
    method='bicubic',
    antialias=True
).astype(dtype)
```

#### ImageEncoderViT class (Lines 77-184)
```python
# Parameter initialization (Line 128):
# CURRENT:
self.pos_embed = nn.Parameter(
    torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
)

# CHANGE TO:
self.pos_embed = mx.zeros([1, img_size // patch_size, img_size // patch_size, embed_dim])
# Note: In MLX, arrays are trainable by default when part of a module
```

#### Attention class (Lines 252-323)
```python
# CURRENT (Lines 310-315):
if self.use_rel_pos:
    rel_h = rel_h.view(B, self.num_heads, rel_h.size(1), rel_h.size(2), rel_h.size(3))
    rel_w = rel_w.view(B, self.num_heads, rel_w.size(1), rel_w.size(2), rel_w.size(3))
    attn_bias = (rel_h + rel_w).view(B, self.num_heads, rel_h.size(2), rel_h.size(3) * rel_w.size(4))
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)

# CHANGE TO:
if self.use_rel_pos:
    rel_h = rel_h.reshape([B, self.num_heads, rel_h.shape[1], rel_h.shape[2], rel_h.shape[3]])
    rel_w = rel_w.reshape([B, self.num_heads, rel_w.shape[1], rel_w.shape[2], rel_w.shape[3]])
    attn_bias = (rel_h + rel_w).reshape([B, self.num_heads, rel_h.shape[2], rel_h.shape[3] * rel_w.shape[4]])
    x = mx.fast.scaled_dot_product_attention(q, k, v, mask=attn_bias)

# Without relative position (Lines 313):
# CURRENT:
x = torch.nn.functional.scaled_dot_product_attention(q, k, v)

# CHANGE TO:
x = mx.fast.scaled_dot_product_attention(q, k, v)
```

### 2. `/DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepencoder/clip_sdpa.py`

**Line-by-Line Changes:**

#### Imports (Lines 1-9)
```python
# CURRENT:
import torch
from torch import nn
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

# CHANGE TO:
import mlx.core as mx
import mlx.nn as nn
# Remove flash_attn imports - will use mx.fast.scaled_dot_product_attention
```

#### get_abs_pos function (Lines 63-99)
```python
# CURRENT (Lines 85-92):
old_pos_embed = old_pos_embed.to(torch.float32)
new_pos_embed = F.interpolate(
    old_pos_embed,
    size=(tgt_size, tgt_size),
    mode='bicubic',
    antialias=True,
    align_corners=False,
).to(dtype)

# CHANGE TO:
old_pos_embed = old_pos_embed.astype(mx.float32)
new_pos_embed = mx.image.resize(
    old_pos_embed,
    [tgt_size, tgt_size],
    method='bicubic',
    antialias=True
).astype(dtype)
```

#### quick_gelu function (Lines 101-103)
```python
# CURRENT:
@torch.jit.script
def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)

# CHANGE TO:
# Remove @torch.jit.script decorator (MLX doesn't use JIT compilation the same way)
def quick_gelu(x):
    return x * mx.sigmoid(1.702 * x)
```

#### NoTPAttention class (Lines 227-284)
```python
# CURRENT (Lines 251-252):
if self.use_flash_attention:
    output = flash_attn_qkvpacked_func(xqkv)

# CHANGE TO:
if self.use_flash_attention:
    # Convert xqkv from [B, seq, 3, heads, head_dim] to separate q, k, v
    q, k, v = mx.split(xqkv, 3, axis=2)
    q = mx.squeeze(q, axis=2)
    k = mx.squeeze(k, axis=2)
    v = mx.squeeze(v, axis=2)
    # Reshape for attention: [B, heads, seq, head_dim]
    q = q.transpose([0, 2, 1, 3])
    k = k.transpose([0, 2, 1, 3])
    v = v.transpose([0, 2, 1, 3])
    output = mx.fast.scaled_dot_product_attention(q, k, v)
    output = output.transpose([0, 2, 1, 3]).reshape([bsz, seqlen, -1])

# CURRENT (Line 281):
output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None)

# CHANGE TO:
output = mx.fast.scaled_dot_product_attention(xq, xk, xv)
```

### 3. `/DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepencoder/build_linear.py`

**Line-by-Line Changes:**

#### Imports (Lines 1-4)
```python
# CURRENT:
import torch.nn as nn
import torch
import torch.nn.functional as F

# CHANGE TO:
import mlx.core as mx
import mlx.nn as nn
```

#### F.unfold usage (Lines 104, 153)
```python
# CURRENT (Line 104):
patches = x.unfold(2, 2, 2).unfold(3, 2, 2)

# CHANGE TO:
# Need to implement unfold functionality manually in MLX
# This extracts 2x2 patches with stride 2
def unfold_2x2(x):
    # x shape: [batch, channels, h, w]
    b, c, h, w = x.shape
    # Reshape to extract 2x2 patches
    x = x.reshape([b, c, h//2, 2, w//2, 2])
    x = x.transpose([0, 1, 2, 4, 3, 5])
    x = x.reshape([b, c, h//2 * w//2, 4])
    return x

patches = unfold_2x2(x)

# CURRENT (Line 153):
x = F.unfold(x, kernel_size=self.cfg.downsample_ratio, stride=self.cfg.downsample_ratio, padding=0)

# CHANGE TO:
# Need custom implementation for general unfold
def unfold_patches(x, kernel_size, stride):
    b, c, h, w = x.shape
    out_h = (h - kernel_size) // stride + 1
    out_w = (w - kernel_size) // stride + 1

    # Extract patches using array slicing
    patches = []
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            w_start = j * stride
            patch = x[:, :, h_start:h_start+kernel_size, w_start:w_start+kernel_size]
            patches.append(patch.reshape([b, c * kernel_size * kernel_size]))

    return mx.stack(patches, axis=2)  # [B, C*kernel*kernel, num_patches]

x = unfold_patches(x, self.cfg.downsample_ratio, self.cfg.downsample_ratio)
```

#### F.pad usage (Line 149)
```python
# CURRENT:
x = F.pad(x, (0, 0, 0, pad, 0, pad), "constant", 0)

# CHANGE TO:
# MLX pad expects [(before, after), ...] for each dimension
x = mx.pad(x, [(0, 0), (0, 0), (0, pad), (0, pad)], constant_values=0)
```

### 4. `/DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py`

**Line-by-Line Changes:**

#### Imports (Lines 1-45)
```python
# CURRENT:
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# CHANGE TO:
import mlx.core as mx
import mlx.nn as nn
# Note: einops operations can be replaced with reshape/transpose
```

#### with torch.no_grad() (Lines 384-466)
```python
# CURRENT:
with torch.no_grad():
    for jdx in range(images_spatial_crop.size(0)):
        # processing code

# CHANGE TO:
# MLX doesn't need explicit no_grad context
# Gradient tracking is controlled differently
for jdx in range(images_spatial_crop.shape[0]):
    # processing code
```

#### Tensor operations throughout
```python
# CURRENT examples:
image_ori.to(torch.bfloat16)
torch.sum(patches).item()
torch.cat([features1, features2], dim=-1)

# CHANGE TO:
image_ori.astype(mx.bfloat16)
mx.sum(patches).item()
mx.concatenate([features1, features2], axis=-1)
```

### 5. `/DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/image_process.py`

**Line-by-Line Changes:**

#### Imports (Lines 1-9)
```python
# CURRENT:
import torch

# CHANGE TO:
import mlx.core as mx
```

#### Tensor creation (Lines 466-497)
```python
# CURRENT:
input_ids = torch.LongTensor(tokenized_str)
target_ids = torch.LongTensor(masked_tokenized_str)
images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)
pixel_values = torch.zeros((1, 3, self.base_size, self.base_size))
pixel_values = torch.stack(images_list, dim=0)
images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
images_crop = torch.stack(images_crop_list, dim=0).unsqueeze(0)

# CHANGE TO:
input_ids = mx.array(tokenized_str, dtype=mx.int64)
target_ids = mx.array(masked_tokenized_str, dtype=mx.int64)
images_seq_mask = mx.array(images_seq_mask, dtype=mx.bool_)
pixel_values = mx.zeros([1, 3, self.base_size, self.base_size])
pixel_values = mx.stack(images_list, axis=0)
images_spatial_crop = mx.array(images_spatial_crop, dtype=mx.int64)
images_crop = mx.stack(images_crop_list, axis=0)
images_crop = mx.expand_dims(images_crop, axis=0)
```

## Critical Implementation Challenges

### 1. Flash Attention Migration
**Challenge:** DeepSeek-OCR heavily uses flash-attention library
**Solution:** Use `mx.fast.scaled_dot_product_attention` (available from v0.28.0)
**Impact:** May have slight performance difference, but MLX's implementation is optimized for Apple Silicon

### 2. F.unfold Operation
**Challenge:** No direct equivalent in MLX
**Solution:** Implement custom unfold using array slicing and reshaping
**Impact:** Moderate - needs careful implementation to match behavior
**Code Required:** ~30-50 lines

### 3. vLLM Integration
**Challenge:** vLLM is CUDA-specific
**Solution:** Replace with MLX-LM or custom MLX inference pipeline
**Impact:** High - requires significant refactoring
**Alternative:** Focus on Transformers-based inference first (simpler path)

### 4. Device Management
**Challenge:** Explicit `.cuda()` calls throughout
**Solution:** MLX automatically uses available device (Metal on Apple Silicon)
**Impact:** Low - mostly removal of device-related code

### 5. Gradient Context
**Challenge:** `torch.no_grad()` contexts
**Solution:** MLX handles gradients differently - may not need explicit contexts
**Impact:** Low - contextual changes

### 6. Parameter Management
**Challenge:** `nn.Parameter` wrapping
**Solution:** MLX modules handle trainable arrays automatically
**Impact:** Low - structural change in module definition

### 7. Image Processing Pipeline
**Challenge:** Integration with PIL and torchvision transforms
**Solution:** Keep PIL, implement transforms in MLX
**Impact:** Low-Medium - mostly works as-is

## Migration Strategy

### Phase 1: Core Operations (Week 1-2)
1. Create MLX versions of vision encoders (SAM, CLIP)
2. Implement custom unfold operation
3. Replace attention mechanisms with MLX equivalents
4. Test individual components

### Phase 2: Integration (Week 2-3)
1. Port MLP projector
2. Integrate vision encoders
3. Update image processing pipeline
4. Test end-to-end vision processing

### Phase 3: Language Model (Week 3-4)
1. Choose inference path: MLX-LM or custom
2. Implement language model integration
3. Handle multi-modal embedding merging
4. Test complete pipeline

### Phase 4: Optimization (Week 4-5)
1. Profile performance
2. Optimize bottlenecks
3. Leverage MLX-specific features
4. Comprehensive testing

### Phase 5: Validation (Week 5-6)
1. Compare outputs with original implementation
2. Benchmark performance
3. Test on various input types
4. Document final implementation

## Expected Performance Characteristics

### Advantages of MLX
1. **Unified Memory:** No CPU-GPU transfers on Apple Silicon
2. **Optimized for M-series:** Native Metal optimization
3. **Automatic Device Management:** Simpler code
4. **Efficient Memory Usage:** Better for large models

### Potential Challenges
1. **Flash Attention:** May not match CUDA flash-attention speed exactly
2. **First Run:** MLX compiles kernels on first run (caching helps)
3. **Maturity:** MLX is newer than PyTorch CUDA stack
4. **Library Ecosystem:** Fewer pre-built tools than CUDA

## File Structure Changes

### New Files to Create
1. `/mlx_port/deepencoder/sam_vary_mlx.py` - MLX version of SAM encoder
2. `/mlx_port/deepencoder/clip_mlx.py` - MLX version of CLIP encoder
3. `/mlx_port/deepencoder/build_linear_mlx.py` - MLX version of projector
4. `/mlx_port/deepencoder/utils.py` - MLX helper functions (unfold, etc.)
5. `/mlx_port/deepseek_ocr_mlx.py` - Main MLX model
6. `/mlx_port/process/image_process_mlx.py` - MLX image processor
7. `/mlx_port/run_dpsk_ocr_mlx.py` - MLX inference script
8. `/mlx_port/requirements_mlx.txt` - MLX dependencies

### Modified Files
1. `config.py` - Add MLX-specific configurations
2. `README.md` - Add MLX installation and usage instructions

## Dependencies Update

### Current Requirements
```
transformers==4.46.3
tokenizers==0.20.3
torch==2.6.0
torchvision==0.21.0
vllm==0.8.5
flash-attn==2.7.3
PyMuPDF
img2pdf
einops
easydict
addict
Pillow
numpy
```

### MLX Requirements
```
mlx>=0.28.0  # For SDPA support
mlx-lm>=0.10.0  # For language model inference
transformers==4.46.3  # Keep for tokenizer
tokenizers==0.20.3
PyMuPDF
img2pdf
easydict
addict
Pillow
numpy
```

## Testing Strategy

### Unit Tests
1. Test each converted component individually
2. Compare tensor operations outputs
3. Verify attention mechanism correctness
4. Check image processing pipeline

### Integration Tests
1. Test vision encoder outputs
2. Verify projector outputs
3. Test complete forward pass
4. Compare with PyTorch outputs (within tolerance)

### Performance Tests
1. Measure inference latency
2. Profile memory usage
3. Compare with CUDA version on same inputs
4. Test different image sizes

## Risk Assessment

### Low Risk
- Basic tensor operations (direct equivalents exist)
- Linear layers, convolutions
- Activation functions
- Image preprocessing

### Medium Risk
- Attention mechanism (different API, should work)
- Custom unfold implementation
- Parameter management differences
- Gradient computation differences

### High Risk
- vLLM integration (no direct equivalent)
- Flash attention performance parity
- Large model memory management
- Multi-GPU support (if needed)

## Conclusion

The migration from DeepSeek-OCR (PyTorch/CUDA) to MLX for Apple Silicon is **highly feasible** given MLX's recent CUDA support (v0.26.0-0.29.3). The major components have clear equivalents:

✅ **Available in MLX:**
- Core tensor operations
- Neural network layers (Linear, Conv2d, LayerNorm)
- Scaled dot-product attention (SDPA)
- Convolution operations
- Image processing primitives
- Softmax, normalization layers

⚠️ **Needs Custom Implementation:**
- F.unfold operation (~50 lines)
- vLLM replacement (use MLX-LM)
- Some specific helper functions

🎯 **Expected Timeline:** 4-6 weeks for full migration
🎯 **Estimated Effort:** Medium-High
🎯 **Success Probability:** High (90%+)

The key to success is:
1. Start with vision encoders (most complex)
2. Use Transformers inference path (simpler than vLLM)
3. Leverage MLX's built-in SDPA instead of custom flash attention
4. Test incrementally at each stage

## Next Steps

1. Create `/mlx_port` directory structure
2. Start with SAM encoder conversion
3. Implement and test attention mechanism
4. Move to CLIP encoder
5. Integrate with simple inference pipeline
6. Iteratively optimize and test

Would you like me to proceed with implementing any specific component?
