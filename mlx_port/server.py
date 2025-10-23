"""
Simple FastAPI server for DeepSeek-OCR MLX web UI.

This server handles image upload and OCR processing for the HTML/CSS/JS UI.

Usage:
    python server.py

Requirements:
    pip install fastapi uvicorn python-multipart Pillow
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import mlx.core as mx
    from models.deepseek_ocr_causal_lm_mlx import DeepseekOCRConfig, build_deepseek_ocr_model
    from preprocessing.image_processor_mlx import DeepseekOCRProcessor
    from inference.pipeline_mlx import DeepSeekOCRPipeline
    MLX_AVAILABLE = True
except ImportError as e:
    MLX_AVAILABLE = False
    IMPORT_ERROR = str(e)


# Create FastAPI app
app = FastAPI(title="DeepSeek-OCR MLX API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (loaded on startup)
pipeline = None
model_mode = "Not Loaded"


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global pipeline, model_mode

    if not MLX_AVAILABLE:
        print(f"WARNING: MLX not available: {IMPORT_ERROR}")
        print("Server will run but OCR functionality will be limited.")
        return

    try:
        print("Loading DeepSeek-OCR model...")

        # Check for weights
        weights_dir = Path(__file__).parent / "weights"
        weights_available = weights_dir.exists() and (weights_dir / "config.json").exists()

        if not weights_available:
            print("WARNING: Model weights not found. Using demo mode.")
            print("Download weights for full functionality.")

            # Create demo model
            config = DeepseekOCRConfig(
                image_token_id=128256,
                n_embed=128,
                vocab_size=102400,
                hidden_size=128,
            )
            model = build_deepseek_ocr_model(config)

            # Create dummy tokenizer
            class DummyTokenizer:
                vocab = {"<image>": 128256, "<pad>": 0}
                bos_token_id = 1
                eos_token_id = 2
                pad_token_id = 0
                def encode(self, text, add_special_tokens=False):
                    return [hash(word) % 1000 + 100 for word in text.split()]
                def decode(self, token_ids, skip_special_tokens=True):
                    return " ".join([f"token_{id}" for id in token_ids[:20]])

            tokenizer = DummyTokenizer()
            model_mode = "Demo Mode"
        else:
            # TODO: Load real model from weights
            config = DeepseekOCRConfig(
                image_token_id=128256,
                n_embed=1280,
                vocab_size=102400,
            )
            model = build_deepseek_ocr_model(config)
            tokenizer = None  # Load real tokenizer
            model_mode = "Full Model"

        # Create processor
        processor = DeepseekOCRProcessor(
            tokenizer,
            image_size=1024,
            base_size=1280,
        )

        # Create pipeline
        pipeline = DeepSeekOCRPipeline(model, processor, tokenizer)

        print(f"Model loaded successfully ({model_mode})")

    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        import traceback
        traceback.print_exc()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "mlx_available": MLX_AVAILABLE,
        "model_loaded": pipeline is not None,
        "model_mode": model_mode,
    }


@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    max_tokens: int = Form(2048),
    temperature: float = Form(0.0),
    top_p: float = Form(0.9),
    cropping: bool = Form(True),
):
    """
    Process an uploaded image with OCR.

    Args:
        file: Uploaded image file
        prompt: Text prompt with <image> placeholder
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        cropping: Enable multi-scale cropping

    Returns:
        JSON with generated text and metadata
    """
    if pipeline is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Model not loaded. Check server logs."}
        )

    try:
        # Read uploaded file
        contents = await file.read()

        # Handle PDF
        if file.content_type == "application/pdf":
            try:
                import fitz  # PyMuPDF

                # Convert first page to image
                pdf_document = fitz.open(stream=contents, filetype="pdf")
                page = pdf_document[0]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom
                img_bytes = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_bytes))

            except ImportError:
                return JSONResponse(
                    status_code=400,
                    content={"error": "PyMuPDF not installed. Install: pip install pymupdf"}
                )
            except Exception as e:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Failed to process PDF: {str(e)}"}
                )
        else:
            # Handle image
            image = Image.open(io.BytesIO(contents))

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Run inference
        result = pipeline.generate(
            images=[image],
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            cropping=cropping,
        )

        return {
            "text": result.get('text', ''),
            "num_tokens": result.get('num_tokens', 0),
            "token_ids": result.get('token_ids', []),
            "model_mode": model_mode,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing failed: {str(e)}"}
        )


@app.post("/stream")
async def stream_generate(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    max_tokens: int = Form(2048),
    temperature: float = Form(0.0),
    top_p: float = Form(0.9),
    cropping: bool = Form(True),
):
    """
    Process an image with streaming output (for future WebSocket support).
    """
    # TODO: Implement streaming with Server-Sent Events or WebSockets
    return JSONResponse(
        status_code=501,
        content={"error": "Streaming not yet implemented. Use /process endpoint."}
    )


if __name__ == "__main__":
    import uvicorn

    print("=" * 80)
    print("DeepSeek-OCR MLX Server")
    print("=" * 80)
    print()
    print("Starting server...")
    print("API docs: http://localhost:8000/docs")
    print("Health check: http://localhost:8000/health")
    print()
    print("For web UI:")
    print("1. Open web_ui/index.html in your browser")
    print("2. Or use Streamlit: streamlit run app.py")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 80)
    print()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
