from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from govdocs_api.utilities.types import TesseactOCRRequest
import json
import cv2
from PIL import Image, ImageEnhance
import pytesseract
import os
import numpy as np
from govdocs_api.utilities.pdf_utilities import MAX_WORKERS, extract_images_from_pdf
import io
from govdocs_api.supabase.db_functions import supabase, create_ocr_request, update_ocr_request_status, get_document_by_barcode, create_ocr_job
from functools import partial
from typing import Dict, Any, List, Optional
import traceback
import asyncio
import threading
import concurrent.futures

# Create a dictionary to store active processing threads
active_requests = {}

def run_in_thread(target_function):
    """Decorator to run a function in a separate thread."""
    def run(*args, **kwargs):
        t = threading.Thread(target=target_function, args=args, kwargs=kwargs)
        t.daemon = True  # Daemonize thread to allow program to exit
        t.start()
        return t
    return run

tesseract = APIRouter()

# DPI = 256
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

def ocr_page(image_tuple : tuple[Image.Image, int], dpi: int, contrast: float) -> dict:
    """Process a single page and return the OCR text."""
    try:
        image, page_num = image_tuple

        # Setting the contrast parameter to 0 will disable the image preprocessor
        if contrast == 0:
            ocr_text = pytesseract.image_to_string(image, lang=LANG, config=f"--dpi {dpi}")
        else:    
            # adjust exposure
            if contrast != 1.0:
                brightness_enhancer = ImageEnhance.Brightness(image)
                brightened_image = brightness_enhancer.enhance(contrast)
                contrast_enhancer = ImageEnhance.Contrast(brightened_image)
                contrasted_image = contrast_enhancer.enhance(contrast)
            else:
                contrasted_image = image
            # remove noise
            denoised_image = remove_bleed_through(np.array(contrasted_image))
            processed_image = Image.fromarray(denoised_image)
            processed_image.info['dpi'] = (dpi, dpi)
            ocr_text = pytesseract.image_to_string(processed_image, lang=LANG, config=f"--dpi {dpi}")

        return {"text": ocr_text, "page_number": page_num}
    except Exception as e:
        print(f"Error in OCR for page {page_num}: {str(e)}")
        return {"text": f"Error: {str(e)}", "page_number": page_num, "error": True}

@run_in_thread
def process_tesseract_request_thread(request_id: int, document_id: str, barcode: str, 
                                   first_page: int, last_page: int, dpi: int, contrast: float):
    """
    Process Tesseract OCR in a separate thread and save results to the database.
    This function runs in its own thread to avoid blocking the FastAPI event loop.
    """
    try:
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the async processing function in this thread's event loop
        loop.run_until_complete(
            _process_ocr_request(
                request_id, document_id, barcode, first_page, last_page, dpi, contrast
            )
        )
    except Exception as e:
        print(f"Thread error processing Tesseract OCR request {request_id}: {str(e)}")
        traceback.print_exc()
        # We need to run the status update in the thread's event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(update_ocr_request_status(request_id, "error"))
    finally:
        # Clean up
        if request_id in active_requests:
            del active_requests[request_id]

async def _process_ocr_request(request_id: int, document_id: str, barcode: str, 
                             first_page: int, last_page: int, dpi: int, contrast: float):
    """
    Process OCR in the background and save results to the database.
    """
    try:
        print(f"Performing OCR on {last_page - first_page + 1} pages for barcode {barcode}")
        
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
                print(f"Error downloading image for barcode {barcode}, page {page_num}: {str(e)}")
                # Create error record in the database
                await create_ocr_job(
                    request_id=request_id,
                    document_id=document_id,
                    page_number=page_num,
                    ocr_output=f"Error: {str(e)}",
                    ocr_model="tesseract",
                    ocr_config={"dpi": dpi, "contrast": contrast},
                    status="error"
                )
                continue
        
        if not page_images_with_numbers:
            print(f"No images could be downloaded for barcode {barcode}")
            await update_ocr_request_status(request_id, "error")
            return
        
        # Perform OCR on the downloaded images
        ocr_config = {"dpi": dpi, "contrast": contrast}
        ocr_with_dpi = partial(ocr_page, dpi=dpi, contrast=contrast)
        
        # Use ThreadPoolExecutor for better stability
        ocr_texts = []
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(page_images_with_numbers))) as executor:
            futures = {executor.submit(ocr_with_dpi, image_tuple): image_tuple for image_tuple in page_images_with_numbers}
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    ocr_texts.append(result)
                except Exception as e:
                    image_tuple = futures[future]
                    page_num = image_tuple[1]
                    error_msg = f"OCR processing failed for page {page_num}: {str(e)}"
                    print(error_msg)
                    traceback.print_exc()
                    ocr_texts.append({
                        "text": error_msg,
                        "page_number": page_num,
                        "error": True
                    })
        
        # Save results to database
        for result in ocr_texts:
            status = "error" if result.get("error", False) else "completed"
            await create_ocr_job(
                request_id=request_id,
                document_id=document_id,
                page_number=result["page_number"],
                ocr_output=result["text"],
                ocr_model="tesseract",
                ocr_config=ocr_config,
                status=status
            )
        
        # Update request status to completed
        await update_ocr_request_status(request_id, "completed")
        print(f"OCR request {request_id} completed successfully")
        
    except Exception as e:
        error_msg = f"Error processing OCR request {request_id}: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        # Update request status to error
        await update_ocr_request_status(request_id, "error")


@tesseract.get("/tesseract")
async def tesseract_ocr_page(barcode: str, dpi: int = 256, 
                            first_page: int = 1, last_page: int = None, contrast : float = 1.0) -> JSONResponse:
    """
    Perform OCR on specific pages of images stored in Supabase using Tesseract.
    
    Args:
        barcode: Barcode identifier for the document
        first_page: First Page number to OCR (1-based index), if None will default to first page
        last_page: Last Page number to OCR (1-based index), if None will process all pages
        dpi: DPI setting for OCR processing
        contrast: Contrast adjustment (0 to disable preprocessing)
    
    Returns:
        JSON response with request ID and status
    """
    
    if (contrast < 0.7 or contrast > 1.3) and contrast != 0:
         raise HTTPException(status_code=400, detail="Contrast must be within the range 0.7 and 1.3 or set to 0 to disable preprocessing")
    
    try:
        # Get document from database
        document = await get_document_by_barcode(barcode)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document with barcode {barcode} not found")
        
        document_id = document["id"]
        
        # Get document page count if first_page or last_page is not specified
        if first_page is None or last_page is None:
            total_page_count = await get_document_page_count(barcode)
            if total_page_count == 0:
                raise HTTPException(status_code=404, detail=f"No pages found for document with barcode {barcode}")
            
            # Set default values if not specified
            if first_page is None:
                first_page = 1
            if last_page is None:
                last_page = total_page_count
        
        # Input validation
        if last_page < first_page:
            raise HTTPException(status_code=400, detail="Last page number must be greater than or equal to first page number.")
        
        page_range = f"{first_page}-{last_page}"
        ocr_config = {"dpi": dpi, "contrast": contrast}
        
        # Create request record
        request_record = await create_ocr_request(
            document_id=document_id,
            page_range=page_range,
            ocr_model="tesseract",
            ocr_config=ocr_config
        )
        
        if not request_record:
            raise HTTPException(status_code=500, detail="Failed to create OCR request")
        
        request_id = request_record["id"]
        
        # Start processing in a separate thread instead of using BackgroundTasks
        thread = process_tesseract_request_thread(
            request_id=request_id,
            document_id=document_id,
            barcode=barcode,
            first_page=first_page,
            last_page=last_page,
            dpi=dpi,
            contrast=contrast
        )
        
        # Store the thread reference
        active_requests[request_id] = thread
        
        return JSONResponse(content={
            "message": "OCR processing started",
            "request_id": request_id,
            "status": "processing",
            "document_id": document_id,
            "page_range": page_range
        })
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing OCR: {str(e)}")

@tesseract.get("/tesseract/status/{request_id}")
async def tesseract_status(request_id: int) -> JSONResponse:
    """
    Get the status of an OCR request.
    
    Args:
        request_id: ID of the OCR request
        
    Returns:
        JSON response with request status and completed pages
    """
    try:
        # Get request from database
        request = supabase.table("ocr_requests").select("*").eq("id", request_id).execute()
        
        if not request.data or len(request.data) == 0:
            raise HTTPException(status_code=404, detail=f"OCR request with ID {request_id} not found")
        
        request_data = request.data[0]
        
        # Get jobs from database
        jobs = supabase.table("ocr_jobs").select("*").eq("request_id", request_id).execute()
        jobs_data = jobs.data if jobs.data else []
        
        return JSONResponse(content={
            "request_id": request_id,
            "status": request_data["status"],
            "document_id": request_data["document_id"],
            "page_range": request_data["page_range"],
            "completed_pages": len(jobs_data),
            "jobs": jobs_data
        })
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting OCR status: {str(e)}")

@tesseract.get("/tesseract/result/{request_id}")
async def tesseract_result(request_id: int) -> JSONResponse:
    """
    Get the results of a completed OCR request.
    
    Args:
        request_id: ID of the OCR request
        
    Returns:
        JSON response with OCR results
    """
    try:
        # Get request from database
        request = supabase.table("ocr_requests").select("*").eq("id", request_id).execute()
        
        if not request.data or len(request.data) == 0:
            raise HTTPException(status_code=404, detail=f"OCR request with ID {request_id} not found")
        
        request_data = request.data[0]
        
        if request_data["status"] != "completed":
            return JSONResponse(content={
                "request_id": request_id,
                "status": request_data["status"],
                "message": "OCR processing is not yet complete"
            })
        
        # Get jobs from database
        #jobs = supabase.table("ocr_jobs").select("*").eq("request_id", request_id).order("page_number", {"ascending": True}).execute()
        jobs = supabase.table("ocr_jobs").select("*").eq("request_id", request_id).order("page_number", desc=False).execute()
        
        if not jobs.data:
            raise HTTPException(status_code=404, detail=f"No OCR jobs found for request ID {request_id}")
        
        # Format results
        results = [{"text": job["ocr_output"], "page_number": job["page_number"]} for job in jobs.data]
        
        return JSONResponse(content=results)
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting OCR results: {str(e)}")
