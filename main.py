import streamlit as st
from PIL import Image
import tempfile
from text_extraction.extract_text import detect_and_crop_paper, detect_and_crop_text_regions, perform_ocr_on_texts
from text_extraction.nlp_processes import fuzzy_match_with_hidden_char_removal

def main(image_path, input_data):
    # Step 1: Detect and crop paper
    cropped_paper = detect_and_crop_paper(image_path)

    if cropped_paper:
        # Step 2: Detect and crop text regions
        text_images = detect_and_crop_text_regions(cropped_paper)

        # Step 3: Perform OCR on text images
        extracted_texts = perform_ocr_on_texts(text_images)
        return cropped_paper, text_images, extracted_texts

    else:
        st.error("No paper detected or an error occurred during detection.")
        return None, None, None

# Streamlit interface
st.title("Text Extraction and Validation from Paper")
st.write("First, enter the details you wish to validate, then upload an image for text extraction.")

# Get user input for validation data
st.write("Enter details to validate against extracted text:")
name = st.text_input("Name", "")
date = st.text_input("Date (e.g., 25-10-2020)", "")
brand = st.text_input("Brand", "")

# Check if all fields are filled before proceeding with image upload
if name and date and brand:
    input_data = {'name': name, 'date': date, 'brand': brand}
    
    # File uploader
    uploaded_image = st.file_uploader("Now, upload an image...", type=["jpg", "jpeg", "png"])

    # Check if an image is uploaded
    if uploaded_image is not None:
        # Convert uploaded file to PIL Image and display it
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Save the image temporarily and pass the file path to main
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpeg") as temp_file:
            image.save(temp_file.name)
            image_path = temp_file.name

        # Process image to extract cropped paper, text images, and extracted texts
        cropped_paper, text_images, extracted_texts = main(image_path, input_data=None)

        if cropped_paper:
            st.image(cropped_paper, caption="Cropped Paper", use_column_width=True)

            # Display each text image alongside its extracted text
            st.write("Detected Text Regions:")
            for text_img, text in zip(text_images, extracted_texts):
                st.image(text_img, caption=f"Extracted Text: {text}", use_column_width=True)

            # Validate the extracted text data against user input
            if st.button("Validate Text Data"):
                validated_info = fuzzy_match_with_hidden_char_removal(extracted_texts, input_data)
                st.write("Validation Results:", validated_info)
else:
    st.write("Please fill in all the details to proceed.")
