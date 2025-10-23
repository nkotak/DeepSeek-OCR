"""
DeepSeek-OCR MLX Inference Script

Same interface as the original, but using MLX instead of CUDA.

Usage:
    python run_dpsk_ocr.py
"""

from deepseek_ocr_mlx import DeepSeekOCR
import os

# No CUDA needed for MLX!
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

model_name = 'deepseek-ai/DeepSeek-OCR'

print("Loading model...")
model = DeepSeekOCR.from_pretrained(model_name)

# prompt = "<image>\nFree OCR. "
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'your_image.jpg'
output_path = 'your/output/dir'

# Resolution modes:
# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False
# Gundam: base_size = 1024, image_size = 640, crop_mode = True

print("Running inference...")
result = model.infer(
    prompt=prompt,
    image_file=image_file,
    output_path=output_path,
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=True,
    test_compress=True
)

print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)
print(result)
