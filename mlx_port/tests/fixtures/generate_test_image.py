#!/usr/bin/env python3
"""Generate synthetic test images for pytest fixtures"""
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path


def generate_test_image(output_path: Path, size: tuple = (224, 224)):
    """
    Generate a synthetic test image with text and shapes.

    Args:
        output_path: Path to save the image
        size: Image size (width, height)
    """
    # Create a new image with white background
    img = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw some colored rectangles
    draw.rectangle([10, 10, 100, 100], fill=(255, 0, 0), outline=(0, 0, 0), width=2)
    draw.rectangle([120, 10, 210, 100], fill=(0, 255, 0), outline=(0, 0, 0), width=2)
    draw.rectangle([10, 120, 100, 210], fill=(0, 0, 255), outline=(0, 0, 0), width=2)
    draw.rectangle([120, 120, 210, 210], fill=(255, 255, 0), outline=(0, 0, 0), width=2)

    # Draw some circles
    draw.ellipse([20, 20, 90, 90], fill=(128, 0, 128), outline=(255, 255, 255), width=2)
    draw.ellipse([130, 130, 200, 200], fill=(0, 128, 128), outline=(255, 255, 255), width=2)

    # Draw some lines
    for i in range(0, size[0], 30):
        draw.line([(i, 0), (i, size[1])], fill=(200, 200, 200), width=1)
    for i in range(0, size[1], 30):
        draw.line([(0, i), (size[0], i)], fill=(200, 200, 200), width=1)

    # Add text (use default font)
    text = "Test Image"
    try:
        font = ImageFont.load_default()
        # Calculate text position for centering
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (size[0] - text_width) // 2
        text_y = size[1] - 30

        # Draw text with background
        draw.rectangle([text_x - 5, text_y - 5, text_x + text_width + 5, text_y + text_height + 5],
                      fill=(255, 255, 255))
        draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
    except Exception:
        # If font fails, skip text
        pass

    # Save the image
    img.save(output_path, format='JPEG', quality=95)
    print(f"Generated test image: {output_path} ({size[0]}x{size[1]})")
    return img


def generate_document_image(output_path: Path, size: tuple = (1024, 1024)):
    """
    Generate a synthetic document image with text-like patterns.

    Args:
        output_path: Path to save the image
        size: Image size (width, height)
    """
    # Create a new image with white background
    img = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw document-like structure
    # Title area
    draw.rectangle([50, 50, size[0] - 50, 100], fill=(240, 240, 240), outline=(0, 0, 0), width=2)

    # Paragraphs (simulated as gray rectangles)
    y_pos = 130
    for _ in range(8):
        # Vary line lengths to simulate text
        line_length = np.random.randint(size[0] - 200, size[0] - 100)
        draw.rectangle([80, y_pos, line_length, y_pos + 15], fill=(50, 50, 50))
        y_pos += 25

        # Sometimes add a short line (like end of paragraph)
        if np.random.random() > 0.7:
            short_length = np.random.randint(100, 300)
            draw.rectangle([80, y_pos, 80 + short_length, y_pos + 15], fill=(50, 50, 50))
            y_pos += 40

    # Add some boxes (like images or tables in a document)
    if size[0] > 500:
        draw.rectangle([100, 450, 450, 650], fill=(220, 220, 255), outline=(0, 0, 0), width=2)
        draw.line([(100, 550), (450, 550)], fill=(0, 0, 0), width=2)
        draw.line([(275, 450), (275, 650)], fill=(0, 0, 0), width=2)

    # Add margins
    draw.rectangle([0, 0, size[0] - 1, size[1] - 1], outline=(200, 200, 200), width=1)

    # Save the image
    img.save(output_path, format='JPEG', quality=95)
    print(f"Generated document image: {output_path} ({size[0]}x{size[1]})")
    return img


def generate_random_noise_image(output_path: Path, size: tuple = (640, 640)):
    """
    Generate a random noise image for stress testing.

    Args:
        output_path: Path to save the image
        size: Image size (width, height)
    """
    # Generate random noise
    noise = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(noise)

    # Save the image
    img.save(output_path, format='JPEG', quality=95)
    print(f"Generated noise image: {output_path} ({size[0]}x{size[1]})")
    return img


if __name__ == "__main__":
    # Get the fixtures directory
    fixtures_dir = Path(__file__).parent

    # Generate test images
    print("Generating test fixture images...")
    print("=" * 60)

    # Basic test image (224x224)
    generate_test_image(fixtures_dir / "test_image.jpg", size=(224, 224))

    # Medium test image (640x640)
    generate_test_image(fixtures_dir / "test_image_medium.jpg", size=(640, 640))

    # Large test image (1024x1024)
    generate_test_image(fixtures_dir / "test_image_large.jpg", size=(1024, 1024))

    # Document-like image
    generate_document_image(fixtures_dir / "test_document.jpg", size=(1024, 1024))

    # Random noise image
    generate_random_noise_image(fixtures_dir / "test_noise.jpg", size=(640, 640))

    print("=" * 60)
    print("✅ All test images generated successfully!")
