from concurrent.futures import ProcessPoolExecutor
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from govdocs_api.utilities.types import TesseactOCRRequest
import json
import cv2
from PIL import Image, ImageEnhance
import pytesseract
import os
import numpy as np
from govdocs_api.utilities.pdf_utilities import MAX_WORKERS,  extract_images_from_pdf
import io
from govdocs_api.supabase.db_functions import supabase



tesseract = APIRouter()

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

def ocr_page(image_tuple : tuple[Image.Image, int]) -> dict:
    image, page_num = image_tuple
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
    return {"text": ocr_text, "page_number": page_num}


@tesseract.get("/tesseract")
async def tesseract_ocr_page(barcode: str, dpi: int = 256, first_page: int = 1, last_page: int = None) -> JSONResponse:
    """
    Perform OCR on specific pages of images stored in Supabase using Tesseract.
    
    Args:
        barcode: Barcode identifier for the document
        first_page: First Page number to OCR (1-based index)
        last_page: Last Page number to OCR (1-based index)
        dpi: DPI setting for OCR processing
    
    Returns:
        JSON response with OCR'd text for the specified page(s)
    """
    
    # Update the global DPI setting
    global DPI
    DPI = dpi
    
    # Input validation
    if last_page is not None and last_page < first_page:
        raise HTTPException(status_code=400, detail="Last page number must be greater than or equal to first page number.")
    
    # If last_page is not specified, just process the first_page
    if last_page is None:
        last_page = first_page
    
    try:
        # Process each page in the range
        page_images_with_numbers = []
        for page_num in range(first_page, last_page + 1):
            try:
                # Download image from Supabase Storage
                response = (
                    supabase.storage
                    .from_("ia_bucket")
                    .download(f"{barcode}/{page_num}.png")
                )

                print(f"Downloaded image for barcode {barcode}, page {page_num}")
                
                # Convert response to PIL Image
                image = Image.open(io.BytesIO(response))
                page_images_with_numbers.append((image, page_num))
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"Could not find image for barcode {barcode}, page {page_num}: {str(e)}")
        
        # Perform OCR on the downloaded images
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            ocr_texts = list(executor.map(ocr_page, page_images_with_numbers))
        
        ocr_texts.sort(key=lambda x: x['page_number'])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing OCR: {str(e)}")
    
    return JSONResponse(content=ocr_texts)



# @tesseract.get("/tesseract")
# async def tesseract_ocr_page(pdf_path: str, dpi : int = 256, first_page: int = 1, last_page: int = None) -> JSONResponse:
#     """
#     Perform OCR on a specific page of the given PDF using Tesseract.
    
#     Args:
#         pdf_path: Path to the PDF file
#         first_page: First Page number to OCR (1-based index)
#         last_page: Last Page number to OCR (1-based index)
    
#     Returns:
#         JSON response with OCR'd text for the specified page(s)
#     """
#     try:
#         script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#         pdf_path = os.path.join(script_dir, "pdfs", pdf_path)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail="Error rendering PDF: " + str(e) )

#     if (last_page < first_page):
#         raise HTTPException(status_code=400, detail="Last page number must be greater than or equal to first page number.")
    
#     try:
#         images = extract_images_from_pdf(filepath=pdf_path, dpi=dpi, first_page=first_page, last_page=last_page)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail="Error extracting images from PDF: " + str(e) )
    
#     # Perform OCR on the pages
#     with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         ocr_texts = list(executor.map(ocr_page, images))
    
#     ocr_texts.sort(key=lambda x: x['page_number'])
    
#     return JSONResponse(content=ocr_texts)
