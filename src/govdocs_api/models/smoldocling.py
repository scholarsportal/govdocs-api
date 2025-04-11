from contextlib import asynccontextmanager
import gc
from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
import torch
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from pathlib import Path
import os
import time
import base64
from io import BytesIO
from PIL import Image
import json
from typing import List, Dict, Any, Optional
from functools import partial
import concurrent.futures
from govdocs_api.utilities.pdf_utilities import render_pdf_to_base64png, total_pages
from govdocs_api.supabase.db_functions import supabase, create_ocr_request, update_ocr_request_status, get_document_by_barcode, create_ocr_job
import io

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model
    
    processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
    model = AutoModelForVision2Seq.from_pretrained(
        "ds4sd/SmolDocling-256M-preview",
        torch_dtype=torch.bfloat16
    ).to(DEVICE)

    print("SmolDocling model loaded ✅")
    
    yield

    del model
    del processor
    torch.cuda.empty_cache()
    gc.collect()

smoldocling = APIRouter(lifespan=lifespan)

def process_page(page_num : int,  barcode: int, target_longest_image_dim: int =1024, max_new_tokens : int =8192):
    """Process a single page and return the markdown text with performance metrics."""
    perf_metrics = {}
    total_start = time.perf_counter()

    #print(f"Rendering page_num: {page_num} of {pdf_path}")
    
    # Render page to an image
    render_start = time.perf_counter()
    # image_base64 = render_pdf_to_base64png(pdf_path, page_num, target_longest_image_dim=target_longest_image_dim)
    # image_bytes = base64.b64decode(image_base64)
    # image = Image.open(BytesIO(image_bytes))
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
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not find image for barcode {barcode}, page {page_num}: {str(e)}")
    
    render_end = time.perf_counter()
    perf_metrics["download_time"] = render_end - render_start
    
    # Build the messages
    prep_start = time.perf_counter()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Convert this page to docling."}
            ]
        },
    ]
    
    # This fixes the `resolution_max_side` cannot be larger than `max_image_size` error
    custom_size = {"longest_edge": target_longest_image_dim}
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt",images_kwargs={'size': custom_size})
    inputs = inputs.to(DEVICE)
    prep_end = time.perf_counter()
    perf_metrics["preprocessing_time"] = prep_end - prep_start
    
    # Generate outputs
    inference_start = time.perf_counter()
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    inference_end = time.perf_counter()
    perf_metrics["inference_time"] = inference_end - inference_start
    
    # Process outputs
    postproc_start = time.perf_counter()
    prompt_length = inputs.input_ids.shape[1]
    trimmed_generated_ids = generated_ids[:, prompt_length:]
    doctags = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=False,
    )[0].lstrip()
    
    # Create Docling document
    try:
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
        doc = DoclingDocument(name=f"Page {page_num}")
        doc.load_from_doctags(doctags_doc)
        markdown_text = doc.export_to_markdown()
    except Exception as e:
        markdown_text = f"Error processing page: {str(e)}\n\nRaw doctags:\n{doctags}"
    
    postproc_end = time.perf_counter()
    perf_metrics["postprocessing_time"] = postproc_end - postproc_start
    
    total_end = time.perf_counter()
    perf_metrics["total_time"] = total_end - total_start
    
    return {
        "page_number": page_num, 
        "markdown": markdown_text,
        "raw_doctags": doctags,
        "performance": perf_metrics
    }

async def process_smoldocling_request(request_id: int, document_id: str, barcode: str, 
                                     first_page: int, last_page: int, 
                                     target_image_dim: int, max_new_tokens: int,
                                     max_pages: int):
    """
    Process SmolDocling OCR in the background and save results to the database.
    """
    try:
        # Adjust page range
        last_page = min(last_page, first_page + max_pages - 1)
        pages_to_process = range(first_page, last_page + 1)
        
        # Process pages
        results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(pages_to_process))) as executor:
            # Create a partial function with fixed parameters
            process_func = partial(
                process_page,
                barcode=barcode,
                target_longest_image_dim=target_image_dim,
                max_new_tokens=max_new_tokens
            )
            
            # Process selected pages in parallel
            future_to_page = {executor.submit(process_func, page_num): page_num for page_num in pages_to_process}
            
            for future in concurrent.futures.as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"Page {page_num} processed successfully")
                    
                    # Save result to database
                    ocr_config = {
                        "target_image_dim": target_image_dim,
                        "max_new_tokens": max_new_tokens
                    }
                    
                    await create_ocr_job(
                        request_id=request_id,
                        document_id=document_id,
                        page_number=page_num,
                        ocr_output=json.dumps({
                            "markdown": result["markdown"],
                            "raw_doctags": result["raw_doctags"]
                        }),
                        ocr_model="smoldocling",
                        ocr_config=ocr_config
                    )
                    
                except Exception as e:
                    print(f"Error processing page {page_num}: {str(e)}")
                    error_message = f"Error processing page {page_num}: {str(e)}"
                    
                    # Save error to database
                    await create_ocr_job(
                        request_id=request_id,
                        document_id=document_id,
                        page_number=page_num,
                        ocr_output=json.dumps({
                            "markdown": error_message,
                            "raw_doctags": ""
                        }),
                        ocr_model="smoldocling",
                        ocr_config={
                            "target_image_dim": target_image_dim,
                            "max_new_tokens": max_new_tokens
                        },
                        status="error"
                    )
                    
                    results.append({
                        "page_number": page_num,
                        "markdown": error_message,
                        "raw_doctags": "",
                        "performance": {"error": str(e)}
                    })
        
        # Combine all markdown into one document with page markers
        combined_markdown = ""
        for result in sorted(results, key=lambda x: x["page_number"]):
            combined_markdown += f"\n\n## Page {result['page_number']}\n\n{result['markdown']}"
        
        # Update request status to completed
        await update_ocr_request_status(request_id, "completed")
        
    except Exception as e:
        print(f"Error processing SmolDocling request {request_id}: {str(e)}")
        # Update request status to error
        await update_ocr_request_status(request_id, "error")

@smoldocling.get("/smoldocling")
def smoldocling_ocr(
    barcode: str, 
    background_tasks: BackgroundTasks,
    last_page: int, 
    first_page: int = 1, 
    max_pages: int = 3,
    target_image_dim: int = 1024, 
    max_new_tokens: int = 8192
):
    """
    Perform OCR on a PDF using SmolDocling and return markdown text.
    
    Args:
        barcode: Barcode identifier for the document
        first_page: First page number to OCR (1-based index)
        last_page: Last page number to OCR (1-based index)
        max_pages: Maximum number of pages to process
        target_image_dim: Target longest dimension for rendered images
        max_new_tokens: Maximum number of tokens to generate
    
    Returns:
        JSON response with request ID and status
    """
    # Input validation
    if last_page < first_page:
        raise HTTPException(status_code=400, detail="Last page number must be greater than or equal to first page number.")
    
    try:
        # Get document from database
        document = get_document_by_barcode(barcode)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document with barcode {barcode} not found")
        
        document_id = document["id"]
        page_range = f"{first_page}-{last_page}"
        ocr_config = {
            "target_image_dim": target_image_dim,
            "max_new_tokens": max_new_tokens,
            "max_pages": max_pages
        }
        
        # Create request record
        request_record = create_ocr_request(
            document_id=document_id,
            page_range=page_range,
            ocr_model="smoldocling",
            ocr_config=ocr_config
        )
        
        if not request_record:
            raise HTTPException(status_code=500, detail="Failed to create OCR request")
        
        request_id = request_record["id"]
        
        # Add task to background jobs
        background_tasks.add_task(
            process_smoldocling_request,
            request_id=request_id,
            document_id=document_id,
            barcode=barcode,
            first_page=first_page,
            last_page=last_page,
            target_image_dim=target_image_dim,
            max_new_tokens=max_new_tokens,
            max_pages=max_pages
        )
        
        return {
            "message": "SmolDocling OCR processing started",
            "request_id": request_id,
            "status": "processing",
            "document_id": document_id,
            "page_range": page_range
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing OCR: {str(e)}")

@smoldocling.get("/smoldocling/status/{request_id}")
async def smoldocling_status(request_id: int):
    """
    Get the status of a SmolDocling OCR request.
    
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

@smoldocling.get("/smoldocling/result/{request_id}")
async def smoldocling_result(request_id: int):
    """
    Get the results of a completed SmolDocling OCR request.
    
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
        results = []
        combined_markdown = ""
        
        for job in jobs.data:
            try:
                ocr_output = json.loads(job["ocr_output"])
                markdown = ocr_output.get("markdown", "")
                raw_doctags = ocr_output.get("raw_doctags", "")
                
                results.append({
                    "page_number": job["page_number"],
                    "markdown": markdown,
                    "raw_doctags": raw_doctags
                })
                
                combined_markdown += f"\n\n## Page {job['page_number']}\n\n{markdown}"
            except Exception as e:
                results.append({
                    "page_number": job["page_number"],
                    "markdown": f"Error parsing output: {str(e)}",
                    "raw_doctags": ""
                })
        
        return {
            "combined_markdown": combined_markdown.strip(),
            "pages": results
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting OCR results: {str(e)}")

@smoldocling.get("/smoldocling/image")
def smoldocling_image_ocr(
    image_url: str,
    max_new_tokens: int = 8192
):
    """
    Perform OCR on an image using SmolDocling and return markdown text.
    
    Args:
        image_url: URL of the image to process (can be data URL)
        max_new_tokens: Maximum number of tokens to generate
    
    Returns:
        JSON response with markdown text for the image
    """
    api_start = time.perf_counter()
    
    # Load image
    image_load_start = time.perf_counter()
    try:
        if image_url.startswith('data:image'):
            # Handle data URL
            image_data = image_url.split(',', 1)[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
        else:
            # Handle file path or URL
            image = load_image(image_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading image: {str(e)}")
    image_load_end = time.perf_counter()
    
    # Process image
    process_start = time.perf_counter()
    
    # Build the messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Convert this page to docling."}
            ]
        },
    ]
    
    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to(DEVICE)
    
    # Generate outputs
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    prompt_length = inputs.input_ids.shape[1]
    trimmed_generated_ids = generated_ids[:, prompt_length:]
    doctags = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=False,
    )[0].lstrip()
    
    # Create Docling document
    try:
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
        doc = DoclingDocument(name="Image")
        doc.load_from_doctags(doctags_doc)
        markdown_text = doc.export_to_markdown()
    except Exception as e:
        markdown_text = f"Error processing image: {str(e)}\n\nRaw doctags:\n{doctags}"
    
    process_end = time.perf_counter()
    
    # Calculate performance metrics
    api_end = time.perf_counter()
    performance_summary = {
        "image_loading_time": image_load_end - image_load_start,
        "processing_time": process_end - process_start,
        "total_api_time": api_end - api_start
    }
    
    return {
        "markdown": markdown_text,
        "raw_doctags": doctags,
        "performance_summary": performance_summary
    }

