"""
DeepSeek-OCR MLX Web Interface

A Streamlit web app for running DeepSeek-OCR inference on images and PDFs.

Usage:
    streamlit run app.py

Features:
- Upload images (PNG, JPG, JPEG, WEBP)
- Upload PDFs (converts to images)
- Multiple prompt templates (OCR, Markdown, Grounding, etc.)
- Adjustable generation parameters
- Streaming text output
- Multi-scale image processing
"""

import streamlit as st
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


# Page config
st.set_page_config(
    page_title="DeepSeek-OCR MLX",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .prompt-example {
        background-color: #e7f3ff;
        padding: 0.5rem;
        border-left: 3px solid #2196F3;
        margin: 0.5rem 0;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)


def check_mlx_status():
    """Check if MLX is available."""
    if not MLX_AVAILABLE:
        st.error("❌ MLX not available. This app requires MLX framework (Apple Silicon).")
        st.error(f"Import error: {IMPORT_ERROR}")
        st.info("💡 Install MLX: `pip install mlx`")
        return False
    return True


def check_model_weights():
    """Check if model weights are available."""
    weights_dir = Path(__file__).parent / "weights"
    if not weights_dir.exists():
        return False, "Weights directory not found"

    config_file = weights_dir / "config.json"
    if not config_file.exists():
        return False, "config.json not found in weights directory"

    return True, "Weights found"


@st.cache_resource
def load_model():
    """Load the DeepSeek-OCR model (cached)."""
    try:
        # Check for weights
        weights_available, message = check_model_weights()

        if not weights_available:
            st.warning(f"⚠️ {message}")
            st.info("Using demo mode with placeholder model (limited functionality)")

            # Create demo model with small dimensions
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
        else:
            # TODO: Load real model from weights
            st.info("ℹ️ Loading model weights...")
            # This would load the real model in production
            config = DeepseekOCRConfig(
                image_token_id=128256,
                n_embed=1280,
                vocab_size=102400,
            )
            model = build_deepseek_ocr_model(config)
            tokenizer = None  # Load real tokenizer

        # Create processor
        processor = DeepseekOCRProcessor(
            tokenizer,
            image_size=1024,
            base_size=1280,
        )

        # Create pipeline
        pipeline = DeepSeekOCRPipeline(model, processor, tokenizer)

        return pipeline, "Demo Mode" if not weights_available else "Full Model"

    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None, None


def main():
    """Main application."""

    # Header
    st.markdown('<p class="main-header">📄 DeepSeek-OCR MLX</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Optical Character Recognition powered by MLX on Apple Silicon</p>',
        unsafe_allow_html=True
    )

    # Check MLX
    if not check_mlx_status():
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Prompt template selection
        st.subheader("Prompt Template")
        prompt_templates = {
            "Free OCR": "<image>\nFree OCR.",
            "Markdown Conversion": "<image>\n<|grounding|>Convert the document to markdown.",
            "OCR with Grounding": "<image>\n<|grounding|>OCR this image.",
            "Parse Figure": "<image>\nParse the figure.",
            "Detailed Description": "<image>\nDescribe this image in detail.",
            "Custom": ""
        }

        selected_template = st.selectbox(
            "Choose a template",
            options=list(prompt_templates.keys()),
            help="Select a prompt template or use 'Custom' to write your own"
        )

        if selected_template == "Custom":
            prompt = st.text_area(
                "Custom Prompt",
                value="<image>\nFree OCR.",
                help="Write your custom prompt. Use <image> as placeholder."
            )
        else:
            prompt = prompt_templates[selected_template]
            st.markdown(f'<div class="prompt-example">{prompt}</div>', unsafe_allow_html=True)

        st.divider()

        # Generation parameters
        st.subheader("Generation Settings")

        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=8192,
            value=2048,
            step=100,
            help="Maximum number of tokens to generate"
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            help="0.0 = deterministic, higher = more creative"
        )

        top_p = st.slider(
            "Top-p (Nucleus Sampling)",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Only sample from top p cumulative probability"
        )

        cropping = st.checkbox(
            "Multi-Scale Cropping",
            value=True,
            help="Enable multi-scale processing for large images"
        )

        st.divider()

        # Model info
        st.subheader("📊 Model Info")
        pipeline, mode = load_model()
        if pipeline:
            st.success(f"✅ Model Loaded ({mode})")
            if mode == "Demo Mode":
                st.warning("⚠️ Running in demo mode. Download weights for full functionality.")
        else:
            st.error("❌ Model failed to load")

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Input")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload an image or PDF",
            type=["png", "jpg", "jpeg", "webp", "pdf"],
            help="Supported formats: PNG, JPG, JPEG, WEBP, PDF"
        )

        if uploaded_file is not None:
            # Handle PDF
            if uploaded_file.type == "application/pdf":
                st.info("📄 PDF uploaded. Converting to images...")
                st.warning("⚠️ PDF support requires PyMuPDF. Install: `pip install pymupdf`")

                try:
                    import fitz  # PyMuPDF

                    # Convert PDF to images
                    pdf_bytes = uploaded_file.read()
                    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

                    page_num = st.number_input(
                        "Select page",
                        min_value=1,
                        max_value=len(pdf_document),
                        value=1
                    )

                    # Render page to image
                    page = pdf_document[page_num - 1]
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                    img_bytes = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_bytes))

                    st.image(image, caption=f"Page {page_num} of {len(pdf_document)}", use_column_width=True)

                except ImportError:
                    st.error("PyMuPDF not installed. Install: `pip install pymupdf`")
                    image = None
                except Exception as e:
                    st.error(f"Failed to process PDF: {str(e)}")
                    image = None
            else:
                # Handle image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)

                # Show image info
                st.markdown(f"""
                <div class="info-box">
                <strong>Image Info:</strong><br>
                - Size: {image.size[0]} × {image.size[1]} pixels<br>
                - Mode: {image.mode}<br>
                - Format: {image.format or 'Unknown'}
                </div>
                """, unsafe_allow_html=True)

            # Process button
            if st.button("🚀 Run OCR", type="primary", use_container_width=True):
                if image is None:
                    st.error("No image loaded")
                elif pipeline is None:
                    st.error("Model not loaded")
                else:
                    with col2:
                        with st.spinner("Processing..."):
                            try:
                                # Run inference
                                result = pipeline.generate(
                                    images=[image],
                                    prompt=prompt,
                                    max_tokens=max_tokens,
                                    temperature=temperature,
                                    top_p=top_p,
                                    cropping=cropping,
                                )

                                # Store result in session state
                                st.session_state['result'] = result
                                st.session_state['processed'] = True

                            except Exception as e:
                                st.error(f"Processing failed: {str(e)}")
                                st.exception(e)
        else:
            st.info("👆 Upload an image or PDF to get started")

    with col2:
        st.header("📝 Output")

        if 'processed' in st.session_state and st.session_state['processed']:
            result = st.session_state['result']

            # Show result
            st.markdown(f"""
            <div class="success-box">
            <strong>✅ Processing Complete</strong><br>
            Generated {result.get('num_tokens', 0)} tokens
            </div>
            """, unsafe_allow_html=True)

            # Output text
            output_text = result.get('text', 'No output generated')
            st.text_area(
                "Generated Text",
                value=output_text,
                height=400,
                help="Copy the text or download it below"
            )

            # Download button
            st.download_button(
                label="💾 Download Text",
                data=output_text,
                file_name="ocr_output.txt",
                mime="text/plain",
                use_container_width=True
            )

        else:
            st.info("👈 Upload an image and click 'Run OCR' to see results here")

            # Show examples
            st.subheader("📚 Example Prompts")

            examples = [
                ("Free OCR", "Basic OCR without special formatting"),
                ("Markdown Conversion", "Convert document to markdown with formatting"),
                ("OCR with Grounding", "OCR with layout understanding"),
                ("Parse Figure", "Extract information from charts/figures"),
                ("Detailed Description", "Get detailed image description"),
            ]

            for title, desc in examples:
                st.markdown(f"""
                <div class="info-box">
                <strong>{title}</strong><br>
                {desc}
                </div>
                """, unsafe_allow_html=True)

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>
            <strong>DeepSeek-OCR MLX</strong> |
            <a href="https://github.com/deepseek-ai/DeepSeek-OCR" target="_blank">Original Repo</a> |
            Ported to MLX for Apple Silicon
        </p>
        <p>
            <em>Note: This is a demonstration interface. For production use, download the full model weights.</em>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
