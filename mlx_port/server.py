"""
FastAPI server for DeepSeek-OCR MLX.

Provides REST API for OCR processing.

Usage:
    python server.py

Then open web_ui/index.html in your browser.

Requirements:
    pip install fastapi uvicorn python-multipart mlx huggingface_hub transformers safetensors Pillow
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import sys
from pathlib import Path
import tempfile
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Check MLX
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError as e:
    MLX_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Import our API
try:
    from deepseek_ocr_mlx import DeepSeekOCR
    API_AVAILABLE = True
except ImportError as e:
    API_AVAILABLE = False
    API_ERROR = str(e)


# Create app
app = FastAPI(title="DeepSeek-OCR MLX API", version="1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None
model_error = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model, model_error

    if not MLX_AVAILABLE:
        model_error = f"MLX not available: {IMPORT_ERROR}"
        print(f"ERROR: {model_error}")
        return

    if not API_AVAILABLE:
        model_error = f"API not available: {API_ERROR}"
        print(f"ERROR: {model_error}")
        return

    try:
        print("=" * 80)
        print("Loading DeepSeek-OCR model...")
        print("=" * 80)
        print()
        print("First run will download ~10-15GB from HuggingFace")
        print("This may take 10-15 minutes...")
        print()

        model = DeepSeekOCR.from_pretrained('deepseek-ai/DeepSeek-OCR')

        print()
        print("=" * 80)
        print("✅ Model loaded and ready")
        print("=" * 80)
        print()

    except Exception as e:
        model_error = str(e)
        print(f"ERROR loading model: {model_error}")
        import traceback
        traceback.print_exc()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok" if model is not None else "error",
        "mlx_available": MLX_AVAILABLE,
        "api_available": API_AVAILABLE,
        "model_loaded": model is not None,
        "error": model_error,
    }


@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    prompt: str = Form("<image>\nFree OCR."),
    base_size: int = Form(1024),
    image_size: int = Form(640),
    crop_mode: bool = Form(True),
):
    """
    Process an image with OCR.

    Args:
        file: Uploaded image file
        prompt: Text prompt with <image> placeholder
        base_size: Base resolution (512/640/1024/1280)
        image_size: Crop resolution (512/640/1024/1280)
        crop_mode: Enable multi-scale cropping

    Returns:
        JSON with generated text
    """
    if model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model not loaded: {model_error}"
        )

    try:
        # Read uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Run inference
            result = model.infer(
                prompt=prompt,
                image_file=tmp_path,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
            )
        finally:
            # Clean up temp file
            os.unlink(tmp_path)

        return {
            "text": result,
            "status": "success",
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    print("=" * 80)
    print("DeepSeek-OCR MLX Server")
    print("=" * 80)
    print()
    print("Starting server on http://localhost:8000")
    print()
    print("API endpoints:")
    print("  - GET  /health  - Health check")
    print("  - POST /process - Process image")
    print("  - Docs: http://localhost:8000/docs")
    print()
    print("For web UI:")
    print("  Open web_ui/index.html in your browser")
    print()
    print("Or use Streamlit:")
    print("  streamlit run app.py")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 80)
    print()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
