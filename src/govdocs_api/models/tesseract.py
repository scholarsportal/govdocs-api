from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from govdocs_api.utilities.types import TesseactOCRRequest
import json
import cv2
from PIL import Image, ImageEnhance
import pytesseract
import os
import platform
import numpy as np
from govdocs_api.utilities.pdf_utilities import convert_pdf_to_images, render_pdf_to_base64png
import logging

import base64
from io import BytesIO
from PIL import Image

# base64 to Pillow
def base64_to_pil(base64_str):
  pil_img = base64.b64decode(base64_str)
  pil_img = BytesIO(pil_img)
  pil_img = Image.open(pil_img)
  return pil_img

tesseract = APIRouter()

@tesseract.get("/ocr/v1/tesseract/page")
async def tesseract_ocr_page(pdf_path: str, page_number: int) -> JSONResponse:
    """
    Perform OCR on a specific page of the given PDF using Tesseract.
    
    Args:
        pdf_path: Path to the PDF file
        page_number: Page number to OCR (1-based index)
    
    Returns:
        JSON response with OCR'd text for the specified page
    """
    try:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pdf_path = os.path.join(script_dir, "pdfs", pdf_path)
        #images = [render_pdf_to_base64png(local_pdf_path=pdf_path, page_num=page_number, target_longest_image_dim=1024)]
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error rendering PDF: " + str(e) )


    # Convert PDF pages to images
    images = convert_pdf_to_images(pdf_path, first_page=page_number, last_page=page_number)
    
    # # # Check if the requested page exists
    # if page_number < 1 or page_number > len(images):
    #     raise HTTPException(
    #         status_code=400, 
    #         detail=f"Invalid page number. The PDF has {len(images)} pages."
    #     )
    
    # Get the image for the specified page (adjust for 0-based index)
    #page_image = images[page_number - 1] # When we are OCRing the entire PDF, we extract the page image from the list of images.
    page_image = images[0] # When we are OCRing a single page, we extract the page image from the list of images.
    
    # OCR the page using the existing ocr_page function
    DEBUG = False
    FORCE = False
    DPI = 256
    CONTRAST = 1.1
    LANG = "eng+fra"
    
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
    
    # Perform OCR on the page
    page_text = ocr_page(page_image)
    
    return {
        "page_number": page_number,
        "text": page_text
    }


