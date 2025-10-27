# Running DeepSeek-OCR MLX Locally

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.8+**
- **~20GB free disk space** (for model weights)

## Installation

```bash
# Install dependencies
pip install mlx huggingface_hub transformers safetensors Pillow streamlit fastapi uvicorn python-multipart
```

## Option 1: Python API (Simplest)

```python
from deepseek_ocr_mlx import DeepSeekOCR

# Load model (downloads on first run)
model = DeepSeekOCR.from_pretrained('deepseek-ai/DeepSeek-OCR')

# Run OCR
result = model.infer(
    prompt="<image>\nFree OCR.",
    image_file='your_image.jpg'
)

print(result)
```

Or use the script:

```bash
# Edit run_dpsk_ocr.py to set your image path
python run_dpsk_ocr.py
```

## Option 2: Streamlit Web UI (Recommended)

```bash
cd mlx_port
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

**Features:**
- Upload images via web interface
- Select prompt templates
- Choose resolution modes
- Download results as text

## Option 3: FastAPI + HTML UI

### Start the server:

```bash
cd mlx_port
python server.py
```

Server runs on http://localhost:8000

### Open the web UI:

```bash
# In a browser, open:
file:///path/to/DeepSeek-OCR/mlx_port/web_ui/index.html

# Or with Python:
cd web_ui
python -m http.server 8080
# Then open http://localhost:8080
```

**Features:**
- Clean HTML/CSS/JS interface
- Drag-and-drop file upload
- Real-time processing
- Multiple prompt templates

## First Run

**On first run, the model will download ~10-15GB from HuggingFace.**

This takes 10-15 minutes depending on your connection.

```
Loading DeepSeek-OCR for MLX
================================================================================

Downloading and loading model: deepseek-ai/DeepSeek-OCR
This includes EVERYTHING: vision encoders + language model + weights
(First run may take 10-15 minutes to download ~10-15GB)

Downloading model from HuggingFace...
[Progress bar...]
✅ Model downloaded

Loading weights from safetensors...
✅ Loaded 15000+ weight tensors

Converting weights to MLX format...
✅ Converted to MLX

Loading tokenizer...
✅ Tokenizer loaded

Loading weights into MLX model...
✅ All weights loaded successfully

================================================================================
✅ Model fully loaded and ready for inference
================================================================================
```

**Subsequent runs are instant** (loads from cache).

## Resolution Modes

Choose based on your needs:

| Mode | Speed | Quality | Use Case |
|------|-------|---------|----------|
| Tiny (512×512) | ⚡⚡⚡ | ⭐⭐ | Quick tests |
| Small (640×640) | ⚡⚡ | ⭐⭐⭐ | General use |
| Base (1024×1024) | ⚡ | ⭐⭐⭐⭐ | High quality |
| Large (1280×1280) | 🐌 | ⭐⭐⭐⭐⭐ | Best quality |
| Gundam (1024+640 multi-scale) | 🐌 | ⭐⭐⭐⭐⭐ | Large documents |

## Troubleshooting

### Import errors

```bash
pip install --upgrade mlx huggingface_hub transformers safetensors
```

### MLX not available

Only works on Apple Silicon Macs. Won't work on:
- Intel Macs
- Windows
- Linux

For those platforms, use the original CUDA version.

### Out of memory

Try a smaller resolution mode (Tiny or Small).

### Slow download

The model is large (~10-15GB). First download takes time. Be patient.

### Server not starting

Check if port 8000 is already in use:

```bash
lsof -i :8000
# If something is using it, kill it or change the port in server.py
```

## Testing

Test with a simple image:

```python
from deepseek_ocr_mlx import DeepSeekOCR
from PIL import Image

# Create a simple test image
img = Image.new('RGB', (800, 600), color='white')

# Load model
model = DeepSeekOCR.from_pretrained('deepseek-ai/DeepSeek-OCR')

# Save test image
img.save('test.jpg')

# Run OCR
result = model.infer(
    prompt="<image>\nDescribe this image.",
    image_file='test.jpg'
)

print(result)
```

## Performance

Approximate on M3 Max (128GB):

- **First load:** 10-15 minutes (download)
- **Subsequent loads:** 5-10 seconds (from cache)
- **Inference (Base mode):** ~200ms per image
- **Inference (Large mode):** ~500ms per image

## Next Steps

1. Try the Streamlit UI (easiest): `streamlit run app.py`
2. Test with your own images
3. Experiment with different prompts
4. Try different resolution modes

## Support

- Only works on Apple Silicon Macs
- Requires ~20GB disk space
- Requires internet for first download
- Subsequent runs work offline
