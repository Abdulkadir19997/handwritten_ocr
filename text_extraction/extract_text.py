from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import re
from datetime import datetime

# Load models
model_paper = YOLO('text_extraction/weights/paper_detector.pt')
model_text = YOLO('text_extraction/weights/text_detector.pt')
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
ocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
detected_paper_path='text_extraction/detected_objects/cropped_paper.jpeg'


def detect_and_crop_paper(image_path):
    """Detects and crops the paper from the input image."""
    try:
        results_paper = model_paper.predict(image_path)
        for result in results_paper:
            if result.boxes:  # Check if any boxes were detected
                for box in result.boxes:
                    x1, y1, x2, y2 = [coord.item() for coord in box.xyxy[0]]  # Convert Tensor to Python numbers
                    cropped_paper = Image.open(image_path).crop((x1, y1, x2, y2))
                    cropped_paper = cropped_paper.convert("RGB")  # Convert to RGB mode
                    cropped_paper.save(detected_paper_path)
                    return cropped_paper
        # If no boxes were detected, raise an exception
        raise ValueError("No paper detected in the image.")
    except Exception as e:
        print(f"Error during paper detection: {e}")
        return None


def detect_and_crop_text_regions(cropped_paper):
    """Detects and crops text regions from the cropped paper image."""
    results_text = model_text.predict(cropped_paper)
    text_images = []
    for result in results_text:
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = [coord.item() for coord in box.xyxy[0]]  # Convert Tensor to Python numbers
            cropped_text = cropped_paper.crop((x1, y1, x2, y2))
            cropped_text = cropped_text.convert("RGB")  # Convert to RGB mode
            cropped_text_path = f'text_extraction/detected_objects/text_cropped_{i}.jpeg'
            cropped_text.save(cropped_text_path)
            text_images.append(cropped_text)
    return text_images


def perform_ocr_on_texts(text_images):
    """Performs OCR on cropped text images and returns the extracted text."""
    extracted_texts = []
    for text_image in text_images:
        image = text_image.convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = ocr_model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        extracted_texts.append(generated_text)
    return extracted_texts


# # Example usage
# image_path = 'test_data/extract_text_example.png'
# input_data = {'name': 'Abdulkadir pasa', 'date': '10-10-2024', 'brand': 'Netflix'}

# # Step 1: Detect and crop paper with exception handling
# cropped_paper = detect_and_crop_paper(image_path)

# if cropped_paper:
#     # Step 2: Detect and crop text regions
#     text_images = detect_and_crop_text_regions(cropped_paper)

#     # Step 3: Perform OCR on text images
#     extracted_texts = perform_ocr_on_texts(text_images)
#     print("Extracted Texts:", extracted_texts)

# else:
#     print("No paper detected or an error occurred during detection.")
