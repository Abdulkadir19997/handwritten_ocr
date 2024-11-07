## Handwritten Document OCR and Validation Pipeline

## Introduction

This project leverages a pipeline of machine learning models to automate the detection and recognition of handwritten text in scanned images of documents. Specifically, it uses YOLO (You Only Look Once) object detection models to first locate paper documents within an image, then detect individual handwritten lines within each paper. The text is extracted using Microsoft's TrOCR (Transformer-based OCR) model, and the extracted text is validated against provided input text using fuzzy search techniques.


## Pipeline Overview

This pipeline is designed to detect, extract, and validate handwritten text from images of documents through a sequence of model-driven steps. Below is an overview of each step in the process.

### Step 1: Paper Detection

The first YOLO model in the pipeline is trained specifically to detect entire paper documents within an image. This model scans the input image, identifying bounding boxes around each paper present. Once detected, each paper is cropped out of the original image and passed to the next stage for further processing.

- **Model**: YOLO (custom-trained or fine-tuned on paper/document detection)
- **Output**: Cropped images of individual paper documents

### Step 2: Handwriting Detection

For each cropped paper document, a second YOLO model is applied to detect individual handwritten lines. This detailed detection stage isolates each line of handwritten text, allowing the pipeline to produce clean, line-specific images for more accurate OCR (Optical Character Recognition) in the next step.

- **Model**: YOLO (custom-trained or fine-tuned on handwriting line detection)
- **Output**: Cropped images of each detected handwritten line

### Step 3: Text Extraction with TrOCR

Each isolated line image is then processed by the TrOCR model, a Transformer-based OCR model optimized for handling both printed and handwritten text. The TrOCR model analyzes each line image and outputs the recognized text content, providing a digital representation of the handwritten text.

- **Model**: TrOCR
- **Output**: Recognized text for each line of handwriting

### Step 4: Fuzzy Validation

To ensure the accuracy of the recognized text, a fuzzy search algorithm validates the extracted text against the provided input text. This validation step compares each extracted line with the expected content, using a similarity score to measure how closely they match. Fuzzy search allows for tolerance in matching, useful for handling slight OCR errors or variances in handwriting.

- **Algorithm**: Fuzzy search (e.g., Levenshtein distance or cosine similarity)
- **Output**: Similarity scores for each line, indicating the degree of match between extracted and expected text


## Streamlit Demo

This project includes a Streamlit demo that guides users through the process of text extraction and validation from an uploaded document image. Below are screenshots of the interface, with explanations of each stage in the process.

### 1. Text Extraction and Validation Input Form

![Text Extraction and Validation Input Form](readme_images/Ekran_goruntusu_2024-11-08_003519.png)

In this initial form, users can enter the expected details for validation against the extracted text. The form includes fields for:
- **Name**: Enter the expected name.
- **Date**: Enter the expected date in a `dd-mm-yyyy` format.
- **Brand**: Enter the expected brand name.

After entering these details, users can upload an image file, such as a photo of a document or handwritten paper, to initiate the text extraction process.

### 2. Uploaded Image Preview

![Uploaded Image Preview](readme_images/Ekran_goruntusu_2024-11-08_003640.png)

Once the image is uploaded, it is displayed for user confirmation. This preview ensures that the correct file has been selected before the extraction begins. In this example, the uploaded image shows a person holding a paper with handwritten details.

### 3. Cropped Paper Detection

![Cropped Paper Detection](readme_images/Ekran_goruntusu_2024-11-08_003650.png)

After uploading, the system detects and isolates the region of interest containing the handwritten details. The cropped section shows only the document area, removing extraneous parts of the image, which makes it easier to focus on extracting and recognizing text.

### 4. Detected Text Regions and Extracted Text

![Detected Text Regions and Extracted Text](readme_images/Ekran_goruntusu_2024-11-08_003658.png)

The detected text regions are shown here, highlighting each section of handwritten text recognized by the system. Under each region, the extracted text is displayed, which includes:
- **Date**: Extracted as "10-10-2024"
- **Brand**: Extracted as "Netflix"
- **Name**: Extracted as "Abdulkadir Pasa"

This step provides a visual representation of the detected regions and displays the text recognized in each area.

### 5. Validation Results

![Validation Results](readme_images/Ekran_goruntusu_2024-11-08_003710.png)

After extraction, the system runs a validation process to compare the extracted text with the user-provided inputs. The results are displayed in JSON format, indicating whether each field matched the expected input:
- **"name"**: `true`
- **"date"**: `true`
- **"brand"**: `true`

These results confirm if the extracted text corresponds accurately with the expected information. If any field doesn’t match, it will be flagged as `false`, allowing for quick identification of mismatches.

---

This Streamlit demo provides a user-friendly interface for text extraction and validation from document images, leveraging machine learning models for accurate handwritten text recognition.


## Developer Guide for environment setup

### Step 1: Clone the Repository

First, clone the repository to your local machine using the command:

```bash
git clone https://github.com/Abdulkadir19997/handwritten_ocr.git
```

**Keep the project archtiecture the same:**
```
├── main.py
├── readme.md
├── requirements.txt
├── text_extraction
│   ├── detected_objects
│   │   ├── cropped_paper.jpeg
│   │   ├── text_cropped_0.jpeg
│   │   ├── text_cropped_1.jpeg
│   │   ├── text_cropped_2.jpeg
│   ├── extract_text.py
│   ├── nlp_processes.py
│   ├── weights
│   │   ├── paper_detector.pt
│   │   ├── text_detector.pt
│   ├── __init__.py
├── __init__.py
```

### Step 2: Create Python Environment

Inside the downloaded 'handwritten_ocr' folder, create a Python environment, **I used 3.10.12 version of python**. For example, to create an environment named 'ocr_handwritten', use:

```bash
python -m venv ocr_handwritten
```

### Step 3: Activate Environment

Activate the environment with:

**For Windows**
```bash
.\ocr_handwritten\Scripts\activate
```

**For Linux**
```bash
source ocr_handwritten/bin/activate
```

### Step 4: Install Requirements

After confirming that the ocr_handwritten environment is active, install all necessary libraries from the 'requirements.txt' file:

```bash
pip install -r requirements.txt
```


### Step 5: Run the Streamlit Application

In the active 'ocr_handwritten' environment, run the 'main.py' file with:

```bash
streamlit run main.py
```

### Step 8: Open a New Terminal Session

Open a new terminal inside the 'handwritten_ocr' folder and activate the 'ocr_handwritten' environment again:

**For Winows**
```bash
.\ocr_handwritten\Scripts\activate
```

**For Linux**
```bash
source auto_inpaint/bin/activate
```


## Version
The current version is 1.0. Development is ongoing, and any support or suggestions are welcome. Please reach out to me:
Abdulkadermousabasha7@gmail.com & LinkedIn