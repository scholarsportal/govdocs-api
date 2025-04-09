from fastapi import APIRouter, FastAPI, HTTPException, BackgroundTasks
from transformers import AutoProcessor, AutoModelForImageTextToText
from contextlib import asynccontextmanager
import torch
from govdocs_api.utilities.pdf_utilities import render_pdf_to_base64png, total_pages
import time
from typing import List, Dict, Any, Optional, Tuple
import base64
from io import BytesIO
from PIL import Image
import gc
from govdocs_api.supabase.db_functions import supabase
import io


model = None
processor = None 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@asynccontextmanager
async def lifespan(app: FastAPI):
  global model
  global processor
  processor = AutoProcessor.from_pretrained("reducto/RolmOCR")
  model = AutoModelForImageTextToText.from_pretrained("reducto/RolmOCR")
  torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  print("RolmOCR model loaded ✅")

  yield

  del model
  del processor
  torch.cuda.empty_cache()
  gc.collect()

rolmocr = APIRouter(lifespan=lifespan)

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

    prompt = "Return the plain text representation of this document as if you were reading it naturally.\n"
    
    # Build the full prompt
    prep_start = time.perf_counter()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                {"type": "text", "text": prompt},
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

@rolmocr.get("/olmocr")
def _rolmocr(barcode: str, last_page: int, first_page: int = 1, temprature: float = 0.9, dpi:int = 256, max_new_tokens: int = 5000, num_return_sequences: int = 1):
    """
    Perform OCR on a specific page of the given PDF using rolmOCR.
    
    Args:
        barcode: Barcode identifier for the document
        first_page: First Page number to OCR (1-based index)
        last_page: Last Page number to OCR (1-based index)
        temprature: The value used to control the randomness of the generated text
        max_new_tokens: The maximum number of tokens to generate
        num_return_sequences: The number of sequences to generate
    
    Returns:
        JSON response with OCR'd text for the specified page(s)
    """
    api_start = time.perf_counter()
    
    # Configure processing parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = []

    # Process pages
    processing_start = time.perf_counter()
    for page in range(first_page, last_page + 1):  # num_pages
        page_start = time.perf_counter()
        output.append(process_page(
            page_num=page,
            barcode=barcode, 
            temperature=temprature, 
            dpi=dpi, 
            max_new_tokens=max_new_tokens, 
            num_return_sequences=num_return_sequences,
            device=device
        ))
        page_end = time.perf_counter()
        print(f"Page {page+1} processed in {page_end - page_start:.2f}s")
    
    processing_end = time.perf_counter()
    
    # Sort and prepare results
    output.sort(key=lambda x: x['page_number'])
    
    # Calculate overall performance metrics
    api_end = time.perf_counter()
    
    # Add overall performance metrics
    performance_summary = {
        "processing_time": processing_end - processing_start,
        "total_api_time": api_end - api_start,
        "pages_processed": len(output),
        "avg_page_processing_time": (processing_end - processing_start) / max(1, len(output))
    }
    
    # Return with performance metrics
    return {
        "pages": output,
        "performance_summary": performance_summary
    }