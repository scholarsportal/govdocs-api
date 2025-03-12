from fastapi import FastAPI, HTTPException, Request, Response, status, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import json
from redis_om import get_redis_connection
import os

import torch
import base64
import urllib.request
import pytesseract

from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, AutoModelForImageTextToText

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/olmocr/pdf-to-png")
async def pdf_to_png(filename: str, page_num :Optional[int] = 1):
  try:
    # Add debugging to check actual paths
    print(f"Current working directory: {os.getcwd()}")
    path = os.path.join(".", "pdfs", filename)
    print(f"Attempting to access: {path}")
    print(f"File exists: {os.path.exists(path)}")
    print(f"Rendering PDF: {filename}, page: {page_num}")
    # Render the PDF to a PNG image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")
    pdf_path = os.path.join(script_dir, "pdfs", filename)
    print(f"PDF path: {pdf_path}")
    print(f"File path exists: {os.path.exists(pdf_path)}")
    print("============")
    image = render_pdf_to_base64png(local_pdf_path=pdf_path, page_num=page_num, target_longest_image_dim=1024)
    return JSONResponse(content={"image": image}, status_code=200)
  except Exception as e:
    raise HTTPException(status_code=500, detail="Error rendering PDF: " + str(e) )

@app.get("/")
async def root():
  return JSONResponse(content={"message": "Hello World"}, status_code=200)


@app.get("/healthz")
async def health_check():
  return JSONResponse(content={"status": "healthy"})

# Model initialization
got_ocr_model = None
got_ocr_processor = None

# Redis caching using redis-om

redis_client = get_redis_connection(
    host='localhost',
    port=6379,
    decode_responses=True
)


def cache_key(model: str, image_path: str, **kwargs) -> str:
  """Generate a unique cache key for the given model and image path."""
  params = json.dumps(kwargs, sort_keys=True)
  return f"{model}:{image_path}:{params}"


def get_cached_result(key: str) -> Optional[str]:
  """Get the cached result for the given key."""
  return redis_client.get(key)


def set_cached_result(key: str, result: str, expire_time: int = 60000) -> None:
  """Store the result in the Redis cache with the given expiration key."""
  redis_client.setex(key, expire_time, result)

## Testing redis (delete later)
@app.get("/set_name")
async def set_last_name(first_name: str, last_name: str):
  redis_client.setex(f"{first_name}", 60, last_name)
  redis_client.setex(f"{last_name}", 60, first_name)
  return JSONResponse(content={"message": f"{first_name} {last_name} set successfully"})

@app.get("/last_name")
async def get_last_name(first_name: str):
  last_name = redis_client.get(first_name)
  return JSONResponse(content={"first_name": first_name, "last_name": last_name})

@app.get("/first_name")
async def get_first_name(last_name: str):
  first_name = redis_client.get(last_name)
  return JSONResponse(content={"first_name": first_name, "last_name": last_name})
##


# Helper functions

def get_image_path(image_id: str) -> str:
  """
  Get the image path from MiniIO or local storage.
  For testing, we assume images are in the './images' directory.
  """
  # In production, this would fetch from MiniIO
  local_path = f"./images/{image_id}"
  if not os.path.exists(local_path):
    raise HTTPException(status_code=404, detail="Image not found")
  return local_path

# Pydantic model
# Models for request validation


class OCRRequest(BaseModel):
  image_path: str
  lang: Optional[str] = "eng"
  config: Optional[str] = ""


class GOTOCRRequest(BaseModel):
  image_path: str
  format: Optional[bool] = False
  multi_page: Optional[bool] = False
  crop_to_patches: Optional[bool] = False
  max_patches: Optional[int] = None
  color: Optional[str] = None
  box: Optional[List[int]] = None
  max_new_tokens: Optional[int] = 4096


# Model initializations

# Initialize GOT-OCR-2 model
def initialize_got_ocr_model():
  global got_ocr_model, got_ocr_processor
  if got_ocr_model is None:
    device = "cpu"  # "cuda" if torch.cuda.is_available() else
    got_ocr_model = AutoModelForImageTextToText.from_pretrained(
        "stepfun-ai/GOT-OCR-2",
        device_map=device
    )
    got_ocr_processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2")
  return got_ocr_model, got_ocr_processor


# Pytessearct endpoint
@app.post("/ocr/pytesseract")
async def pytesseract_ocr(request: OCRRequest):
  image_path = get_image_path(request.image_path)

  # Check if the result is cached
  key = cache_key("pytesseract", image_path,
                  lang=request.lang, config=request.config)
  cached_result = get_cached_result(key)

  # Return the cached result if available
  if cached_result:
    return JSONResponse(content={"text": cached_result, cached: True}, status_code=200)

  # Perform OCR
  try:
    text = pytesseract.image_to_string(
        Image.open(image_path),
        lang=request.lang,
        config=request.config
    )

    set_cached_result(key, text)
    return JSONResponse(content={"text": text, cached: False}, status_code=200)
  except Exception as e:
    raise HTTPException(
        status_code=500, detail="Error processing image: " + str(e))


# GOT-OCR-2 endpoint
@app.post("/ocr/got-ocr-2")
async def got_ocr(request: GOTOCRRequest):
  image_path = get_image_path(request.image_path)

  # Check cache
  cache_params = {
      "format": request.format,
      "multi_page": request.multi_page,
      "crop_to_patches": request.crop_to_patches,
      "max_patches": request.max_patches,
      "color": request.color,
      "box": request.box,
      "max_new_tokens": request.max_new_tokens
  }

  key = cache_key("got-ocr-2", image_path, **cache_params)
  cached_result = get_cached_result(key)

  if cached_result:
    return JSONResponse(content={"text": cached_result, "cached": True}, status_code=200)

  # Process with GOT-OCR-2
  try:
    device = "cpu"  # "cuda" if torch.cuda.is_available() else
    model, processor = initialize_got_ocr_model()

    # Handle multi-page case
    if request.multi_page and isinstance(request.image_path, list):
      images = [Image.open(get_image_path(path))
                for path in request.image_path]
      inputs = processor(images, return_tensors="pt",
                         format=request.format, multi_page=request.multi_page)
    else:
      # Handle single image case
      image = Image.open(image_path)
      processor_kwargs = {
          return_tensors: "pt",
          format: request.format,
      }

      # Add additional parameters if provided
      if request.crop_to_patches:
        processor_kwargs["crop_to_patches"] = request.crop_to_patches
        if request.max_patches:
          processor_kwargs["max_patches"] = request.max_patches

      if request.color:
        processor_kwargs["color"] = request.color
      elif request.box:
        processor_kwargs["box"] = request.box

      inputs = processor(image, **processor_kwargs).to(device)

    # Generate text
    generate_ids = model.generate(**inputs,
                                  do_sample=False,
                                  tokenizer=processor.tokenizer,
                                  stop_strings="<|im_end|>",
                                  max_new_tokens=request.max_new_tokens)

    # Decode the generated text
    text = processor.decode(
        generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Cache the result
    set_cached_result(key, text)

    return JSONResponse(content={"text": text, cached: True}, status_code=200)

  except Exception as e:
    raise HTTPException(
        status_code=500, detail="Error processing image: " + str(e))

# Batch processing endpoint for GOT-OCR-2


@app.post("/ocr/got-ocr-2/batch")
async def got_ocr_batch(request: List[GOTOCRRequest], background_tasks: BackgroundTasks):
  try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = get_got_ocr_model()

    images = []
    for req in request:
      image_path = get_image_path(req.image_path)
      images.append(Image.open(image_path))

    inputs = processor(images, return_tensors="pt").to(device)

    generate_ids = model.generate(
        **inputs,
        do_sample=False,
        tokenizer=processor.tokenizer,
        stop_strings="<|im_end|>",
        max_new_tokens=4096,
    )

    results = processor.batch_decode(
        generate_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return JSONResponse(content={"results": results})
  except Exception as e:
    raise HTTPException(
        status_code=500, detail=f"Error processing batch: {str(e)}")

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)