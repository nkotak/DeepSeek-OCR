"""
DeepSeek-OCR MLX Streamlit App

Simple web interface for running OCR on images.

Usage:
    streamlit run app.py

Requirements:
    pip install streamlit mlx huggingface_hub transformers safetensors Pillow
"""

import streamlit as st
from PIL import Image
import io
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Check MLX availability
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError as e:
    MLX_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Page config
st.set_page_config(
    page_title="DeepSeek-OCR MLX",
    page_icon="📄",
    layout="wide",
)

# Header
st.title("📄 DeepSeek-OCR MLX")
st.markdown("Optical Character Recognition powered by MLX on Apple Silicon")
st.markdown("---")

# Check MLX
if not MLX_AVAILABLE:
    st.error("❌ MLX not available. This app requires MLX framework (Apple Silicon).")
    st.error(f"Import error: {IMPORT_ERROR}")
    st.info("Install MLX: `pip install mlx`")
    st.stop()

# Check other requirements
try:
    from deepseek_ocr_mlx import DeepSeekOCR
except ImportError as e:
    st.error(f"❌ Missing dependencies: {e}")
    st.info("Install: `pip install huggingface_hub transformers safetensors`")
    st.stop()


@st.cache_resource
def load_model():
    """Load model (cached across sessions)."""
    try:
        st.info("Loading model... (First run downloads ~10-15GB from HuggingFace)")
        model = DeepSeekOCR.from_pretrained('deepseek-ai/DeepSeek-OCR')
        return model, None
    except Exception as e:
        return None, str(e)


# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    # Prompt templates
    st.subheader("Prompt Template")
    prompt_template = st.selectbox(
        "Select template",
        [
            "Free OCR",
            "Markdown Conversion",
            "OCR with Grounding",
            "Parse Figure",
            "Detailed Description",
        ]
    )

    prompts = {
        "Free OCR": "<image>\nFree OCR.",
        "Markdown Conversion": "<image>\n<|grounding|>Convert the document to markdown.",
        "OCR with Grounding": "<image>\n<|grounding|>OCR this image.",
        "Parse Figure": "<image>\nParse the figure.",
        "Detailed Description": "<image>\nDescribe this image in detail.",
    }

    prompt = prompts[prompt_template]
    st.code(prompt)

    st.divider()

    # Resolution settings
    st.subheader("Resolution")
    resolution_mode = st.selectbox(
        "Mode",
        ["Tiny (Fast)", "Small", "Base", "Large (Best)", "Gundam (Multi-scale)"]
    )

    resolution_configs = {
        "Tiny (Fast)": (512, 512, False),
        "Small": (640, 640, False),
        "Base": (1024, 1024, False),
        "Large (Best)": (1280, 1280, False),
        "Gundam (Multi-scale)": (1024, 640, True),
    }

    base_size, image_size, crop_mode = resolution_configs[resolution_mode]

    st.divider()

    # Model loading
    st.subheader("📊 Model Status")
    with st.spinner("Loading model..."):
        model, error = load_model()

    if model:
        st.success("✅ Model loaded and ready")
    else:
        st.error(f"❌ Failed to load model: {error}")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("📤 Input")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "webp"],
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Show image info
        st.info(f"**Size:** {image.size[0]} × {image.size[1]} pixels")

        # Process button
        if st.button("🚀 Run OCR", type="primary", use_container_width=True):
            if model is None:
                st.error("Model not loaded")
            else:
                with col2:
                    with st.spinner("Processing..."):
                        try:
                            # Save to temp file
                            import tempfile
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                                image.save(tmp.name)
                                tmp_path = tmp.name

                            # Run inference
                            result = model.infer(
                                prompt=prompt,
                                image_file=tmp_path,
                                base_size=base_size,
                                image_size=image_size,
                                crop_mode=crop_mode,
                            )

                            # Clean up temp file
                            import os
                            os.unlink(tmp_path)

                            # Store result
                            st.session_state['result'] = result
                            st.session_state['processed'] = True

                        except Exception as e:
                            st.error(f"Processing failed: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
    else:
        st.info("👆 Upload an image to get started")

with col2:
    st.header("📝 Output")

    if 'processed' in st.session_state and st.session_state['processed']:
        result = st.session_state['result']

        st.success("✅ Processing complete!")

        # Output text
        st.text_area(
            "Generated Text",
            value=result,
            height=400,
        )

        # Download button
        st.download_button(
            label="💾 Download Text",
            data=result,
            file_name="ocr_output.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        st.info("👈 Upload an image and click 'Run OCR'")

        # Show instructions
        with st.expander("📚 How to Use"):
            st.markdown("""
            1. **Upload an image** using the file uploader
            2. **Select a prompt template** in the sidebar
            3. **Choose resolution mode** (higher = better quality but slower)
            4. **Click 'Run OCR'** to process
            5. **Download the result** as a text file

            **First run:** Model will download from HuggingFace (~10-15GB, takes 10-15 minutes)

            **Subsequent runs:** Instant loading from cache
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>DeepSeek-OCR MLX | <a href="https://github.com/deepseek-ai/DeepSeek-OCR">Original Repo</a></p>
</div>
""", unsafe_allow_html=True)
