from fastapi import APIRouter
from fastapi.responses import JSONResponse
from govdocs_api.utilities.types import TesseactOCRRequest
import json
import cv2
from PIL import Image, ImageEnhance
import pytesseract
import os
import platform
import numpy as np
from govdocs_api.utilities.pdf_utilities import convert_pdf_to_images

tesseract = APIRouter()





def tesseract_ocr(pdf_path, output_dir):
    output_file = os.path.join(output_dir, 'tesseract_output.txt')
    os.makedirs(output_dir, exist_ok=True)
    
    DEBUG = False
    FORCE = False
    DPI = 80  # 256 for high quality, 196 for medium quality, 120 for low quality
    CONTRAST = 1.1  # lower than 1.0 to reduce contrast and brightness
    LANG = "eng+fra"
    MAX_WORKERS = 16
    
    def remove_bleed_through(image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply slight non-local means denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 8, 7, 21)
        # Smooth the result for cleaner output
        smoothed = cv2.medianBlur(denoised, 1)
        return smoothed
    
    def ocr_page(image):
        # adjust exposure
        if CONTRAST != 1.0:
            brightness_enhancer = ImageEnhance.Brightness(image)
            brightened_image = brightness_enhancer.enhance(CONTRAST)
            contrast_enhancer = ImageEnhance.Contrast(brightened_image)
            contrasted_image = contrast_enhancer.enhance(CONTRAST)
        else:
            contrasted_image = image
        # remove noise
        denoised_image = remove_bleed_through(np.array(contrasted_image))
        processed_image = Image.fromarray(denoised_image)
        processed_image.info['dpi'] = (DPI, DPI)
        ocr_text = pytesseract.image_to_string(processed_image, lang=LANG, config=f"--dpi {DPI}")
        return ocr_text
    
    # Convert PDF to images
    images = convert_pdf_to_images(pdf_path)
    
    full_text = ""
    for i, img in enumerate(images):
        page_text = ocr_page(img)
        full_text += f"\n\n--- PAGE {i+1} ---\n\n"
        full_text += page_text
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    return full_text

@tesseract.get("/ocr/v1/tesseract")
async def tesseract_ocr(pdf_path: str, config: str = "") -> JSONResponse:
    """Perform OCR on the given image using Tesseract."""
    return {"Recievied ocr request": f"pdf_path: {pdf_path}, config: {config}"}


