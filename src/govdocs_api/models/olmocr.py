from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from govdocs_api.utilities.pdf_utilities import render_pdf_to_base64png, total_pages
import os
import torch
import json
import gc
from io import BytesIO
from PIL import Image
import base64
from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
from functools import partial, cache
import time
from typing import List, Dict, Any, Optional, Tuple

from olmocr.prompts import PageResponse, build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text

from govdocs_api.supabase.db_functions import supabase, create_ocr_request, update_ocr_request_status, get_document_by_barcode, create_ocr_job
import io


model = None
processor = None

#,cache_dir="/local/home/hfurquan/myProjects/Leaderboard/cache"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    global processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "allenai/olmOCR-7B-0225-preview", 
        torch_dtype=torch.bfloat16
    ).eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("OLM OCR model loaded ✅")
    
    yield

    
    del model
    del processor
    torch.cuda.empty_cache()
    gc.collect()

olm_ocr = APIRouter(lifespan=lifespan)


def process_page(page_num, barcode, temperature, dpi, max_new_tokens, num_return_sequences, device):
    """Process a single page and return the OCR text with performance metrics."""
    perf_metrics = {}
    total_start = time.perf_counter()
    
    # Download the image for the page
    render_start = time.perf_counter()
   
    try:
        # Download image from Supabase Storage
        response = (
            supabase.storage
            .from_("ia_bucket")
            .download(f"{barcode}/{page_num}.png")
        )

        print(f"Downloaded image for barcode {barcode}, page {page_num}")

        # Convert response bytes to base64
        image_base64 = base64.b64encode(response).decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not find image for barcode {barcode}, page {page_num}: {str(e)}")
    render_end = time.perf_counter()
    perf_metrics["download_time"] = render_end - render_start

    prompt = "Following is a scanned government document page. Return the text content of the page."
    
    # Build the full prompt
    prep_start = time.perf_counter()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ],
        }
    ]
    
    # Apply the chat template and processor
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    main_image = Image.open(BytesIO(base64.b64decode(image_base64)))
    inputs = processor(
        text=[text],
        images=[main_image],
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for (key, value) in inputs.items()}
    prep_end = time.perf_counter()
    perf_metrics["preprocessing_time"] = prep_end - prep_start
    
    # Generate the output
    inference_start = time.perf_counter()
    output = model.generate(
        **inputs,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        do_sample=True,
    )
    inference_end = time.perf_counter()
    perf_metrics["inference_time"] = inference_end - inference_start
    
    # Decode the output
    postproc_start = time.perf_counter()
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_length:]
    text_output = processor.tokenizer.batch_decode(
        new_tokens, skip_special_tokens=True
    )
    
    try:
        page_text = json.loads(text_output[0])['natural_text']
    except:
        page_text = text_output[0]
    postproc_end = time.perf_counter()
    perf_metrics["postprocessing_time"] = postproc_end - postproc_start
    
    total_end = time.perf_counter()
    perf_metrics["total_time"] = total_end - total_start
    
    return {
        "page_number": page_num, 
        "text": page_text,
        "performance": perf_metrics
    }

async def process_olm_request(request_id: int, document_id: str, barcode: str, 
                            first_page: int, last_page: int, temperature: float, 
                            dpi: int, max_new_tokens: int, num_return_sequences: int):
    """
    Process OLM OCR in the background and save results to the database.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Process each page in the requested range
        output = []
        
        for page in range(first_page, last_page + 1):
            try:
                page_result = process_page(
                    page_num=page,
                    barcode=barcode,
                    temperature=temperature,
                    dpi=dpi,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=num_return_sequences,
                    device=device
                )
                
                output.append(page_result)
                print(f"Page {page} processed successfully")
                
                # Save result to database
                ocr_config = {
                    "temperature": temperature,
                    "dpi": dpi,
                    "max_new_tokens": max_new_tokens,
                    "num_return_sequences": num_return_sequences
                }
                
                await create_ocr_job(
                    request_id=request_id,
                    document_id=document_id,
                    page_number=page,
                    ocr_output=page_result["text"],
                    ocr_model="olmocr",
                    ocr_config=ocr_config
                )
                
            except Exception as e:
                print(f"Error processing page {page}: {str(e)}")
                
                # Save error to database
                await create_ocr_job(
                    request_id=request_id,
                    document_id=document_id,
                    page_number=page,
                    ocr_output=f"Error: {str(e)}",
                    ocr_model="olmocr",
                    ocr_config={
                        "temperature": temperature,
                        "dpi": dpi,
                        "max_new_tokens": max_new_tokens,
                        "num_return_sequences": num_return_sequences
                    },
                    status="error"
                )
        
        # Update request status to completed
        await update_ocr_request_status(request_id, "completed")
        
    except Exception as e:
        print(f"Error processing OLM OCR request {request_id}: {str(e)}")
        # Update request status to error
        await update_ocr_request_status(request_id, "error")

@olm_ocr.get("/olmocr")
async def olm(barcode: str, background_tasks: BackgroundTasks, last_page: int, 
             first_page: int = 1, temperature: float = 0.9, dpi: int = 256, 
             max_new_tokens: int = 5000, num_return_sequences: int = 1):
    """
    Perform OCR on a specific page of the given PDF using olmOCR.
    
    Args:
        barcode: Barcode identifier for the document
        first_page: First Page number to OCR (1-based index)
        last_page: Last Page number to OCR (1-based index)
        temperature: The value used to control the randomness of the generated text
        dpi: The DPI to use for image rendering
        max_new_tokens: The maximum number of tokens to generate
        num_return_sequences: The number of sequences to generate
    
    Returns:
        JSON response with request ID and status
    """
    # Input validation
    if last_page < first_page:
        raise HTTPException(status_code=400, detail="Last page number must be greater than or equal to first page number.")
    
    try:
        # Get document from database
        document = await get_document_by_barcode(barcode)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document with barcode {barcode} not found")
        
        document_id = document["id"]
        page_range = f"{first_page}-{last_page}"
        ocr_config = {
            "temperature": temperature,
            "dpi": dpi,
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": num_return_sequences
        }
        
        # Create request record
        request_record = await create_ocr_request(
            document_id=document_id,
            page_range=page_range,
            ocr_model="olmocr",
            ocr_config=ocr_config
        )
        
        if not request_record:
            raise HTTPException(status_code=500, detail="Failed to create OCR request")
        
        request_id = request_record["id"]
        
        # Add task to background jobs
        background_tasks.add_task(
            process_olm_request,
            request_id=request_id,
            document_id=document_id,
            barcode=barcode,
            first_page=first_page,
            last_page=last_page,
            temperature=temperature,
            dpi=dpi,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences
        )
        
        return {
            "message": "OLM OCR processing started",
            "request_id": request_id,
            "status": "processing",
            "document_id": document_id,
            "page_range": page_range
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing OCR: {str(e)}")

@olm_ocr.get("/olmocr/status/{request_id}")
async def olm_status(request_id: int):
    """
    Get the status of an OLM OCR request.
    
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
        
        return {
            "request_id": request_id,
            "status": request_data["status"],
            "document_id": request_data["document_id"],
            "page_range": request_data["page_range"],
            "completed_pages": len(jobs_data),
            "jobs": jobs_data
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting OCR status: {str(e)}")

@olm_ocr.get("/olmocr/result/{request_id}")
async def olm_result(request_id: int):
    """
    Get the results of a completed OLM OCR request.
    
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
            return {
                "request_id": request_id,
                "status": request_data["status"],
                "message": "OCR processing is not yet complete"
            }
        
        # Get jobs from database
        jobs = supabase.table("ocr_jobs").select("*").eq("request_id", request_id).order("page_number", {"ascending": True}).execute()
        
        if not jobs.data:
            raise HTTPException(status_code=404, detail=f"No OCR jobs found for request ID {request_id}")
        
        # Format results
        pages = [{"text": job["ocr_output"], "page_number": job["page_number"]} for job in jobs.data]
        
        # Calculate overall performance metrics
        performance_summary = {
            "pages_processed": len(pages),
            "status": "completed" 
        }
        
        return {
            "pages": pages,
            "performance_summary": performance_summary
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting OCR results: {str(e)}")