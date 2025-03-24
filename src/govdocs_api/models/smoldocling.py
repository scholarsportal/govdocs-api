from contextlib import asynccontextmanager
import gc
from fastapi import FastAPI, APIRouter, HTTPException
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

def process_page(page_num, pdf_path, target_longest_image_dim=1024, max_new_tokens=8192):
    """Process a single page and return the markdown text with performance metrics."""
    perf_metrics = {}
    total_start = time.perf_counter()

    print(f"Rendering page_num: {page_num} of {pdf_path}")
    
    # Render page to an image
    render_start = time.perf_counter()
    image_base64 = render_pdf_to_base64png(pdf_path, page_num, target_longest_image_dim=target_longest_image_dim)
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_bytes))
    render_end = time.perf_counter()
    perf_metrics["render_time"] = render_end - render_start
    
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
    
    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
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
        doc = DoclingDocument(name=f"Page {page_num+1}")
        doc.load_from_doctags(doctags_doc)
        markdown_text = doc.export_to_markdown()
    except Exception as e:
        markdown_text = f"Error processing page: {str(e)}\n\nRaw doctags:\n{doctags}"
    
    postproc_end = time.perf_counter()
    perf_metrics["postprocessing_time"] = postproc_end - postproc_start
    
    total_end = time.perf_counter()
    perf_metrics["total_time"] = total_end - total_start
    
    return {
        "page_number": page_num + 1, 
        "markdown": markdown_text,
        "raw_doctags": doctags,
        "performance": perf_metrics
    }

@smoldocling.get("/smoldocling")
def smoldocling_ocr(
    pdf_path: str, 
    first_page: int = 1, 
    last_page: int = None, 
    max_pages: int = 3,
    target_image_dim: int = 1024, 
    max_new_tokens: int = 8192
):
    """
    Perform OCR on a PDF using SmolDocling and return markdown text.
    
    Args:
        pdf_path: Path to the PDF file
        first_page: First page number to OCR (1-based index)
        last_page: Last page number to OCR (1-based index)
        max_pages: Maximum number of pages to process
        target_image_dim: Target longest dimension for rendered images
        max_new_tokens: Maximum number of tokens to generate
    
    Returns:
        JSON response with markdown text for the specified page(s)
    """
    api_start = time.perf_counter()
    
    # Locate the PDF file
    pdf_locate_start = time.perf_counter()
    try:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pdf_path = os.path.join(script_dir, "pdfs", pdf_path)
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail=f"PDF file not found: {pdf_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error locating PDF: " + str(e))
    pdf_locate_end = time.perf_counter()
    
    # Get the number of pages in the PDF
    page_count_start = time.perf_counter()
    try:
        num_pages = total_pages(pdf_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error counting PDF pages: " + str(e))
    page_count_end = time.perf_counter()
    
    # Adjust page range
    first_page = max(1, first_page)
    if last_page is None or last_page > num_pages:
        last_page = min(num_pages, first_page + max_pages - 1)
    
    # Convert to 0-based indexing for internal use
    pages_to_process = range(first_page, last_page + 1)

    print(f"pages_to_process: {pages_to_process}")
    print(f"len(pages_to_process): {len(pages_to_process)}")
    
    
    # Process pages
    processing_start = time.perf_counter()
    results = []
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(pages_to_process))) as executor:
        # Create a partial function with fixed parameters
        process_func = partial(
            process_page,
            pdf_path=pdf_path,
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
            except Exception as e:
                print(f"Error processing page {page_num}: {str(e)}")
                results.append({
                    "page_number": page_num ,
                    "markdown": f"Error processing page {page_num}: {str(e)}",
                    "raw_doctags": "",
                    "performance": {"error": str(e)}
                })
    
    processing_end = time.perf_counter()
    
    # Sort results by page number
    results.sort(key=lambda x: x["page_number"])
    
    # Combine all markdown into one document with page markers
    combined_markdown = ""
    for result in results:
        combined_markdown += f"\n\n## Page {result['page_number']}\n\n{result['markdown']}"
    
    # Calculate performance metrics
    api_end = time.perf_counter()
    performance_summary = {
        "pdf_location_time": pdf_locate_end - pdf_locate_start,
        "page_counting_time": page_count_end - page_count_start,
        "processing_time": processing_end - processing_start,
        "total_api_time": api_end - api_start,
        "pages_processed": len(results),
        "avg_page_processing_time": (processing_end - processing_start) / max(1, len(results))
    }
    
    return {
        "combined_markdown": combined_markdown.strip(),
        "pages": results,
        "performance_summary": performance_summary
    }

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

