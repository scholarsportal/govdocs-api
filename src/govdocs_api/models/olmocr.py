from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from govdocs_api.utilities.pdf_utilities import render_pdf_to_base64png, total_pages
from olmocr.prompts import build_openai_silver_data_prompt, build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text
import os
import torch
import json
import gc
from io import BytesIO
from PIL import Image
import base64
from fastapi import FastAPI, APIRouter
from contextlib import asynccontextmanager
from functools import partial

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

import concurrent.futures

def process_page(page_num, pdf_path, temperature, dpi, max_new_tokens, num_return_sequences, device):
    """Process a single page and return the OCR text."""
    # Render page to an image
    image_base64 = render_pdf_to_base64png(pdf_path, page_num, target_longest_image_dim=dpi)
    
    # Get anchor text
    try:
        anchor_text = get_anchor_text(pdf_path, page_num, pdf_engine="pdfreport", target_length=4000)
        prompt = build_finetuning_prompt(anchor_text)
    except:
        prompt = ""
    
    # Build the full prompt
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
    
    # Generate the output
    output = model.generate(
        **inputs,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        do_sample=True,
    )
    
    # Decode the output
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_length:]
    text_output = processor.tokenizer.batch_decode(
        new_tokens, skip_special_tokens=True
    )
    
    try:
        page_text = json.loads(text_output[0])['natural_text']
    except:
        page_text = text_output[0]
    
    return {"page_number": page_num + 1, "text": page_text}

@olm_ocr.get("/olmocr")
def olm(pdf_path: str, first_page: int = 1, last_page: int = None, temprature: float = 0.9, dpi:int = 256, max_new_tokens: int = 5000,num_return_sequences: int = 1):
    """
    Perform OCR on a specific page of the given PDF using Tesseract.
    
    Args:
        pdf_path: Path to the PDF file
        first_page: First Page number to OCR (1-based index)
        last_page: Last Page number to OCR (1-based index)
        temprature: The value used to control the randomness of the generated text
        max_new_tokens: The maximum number of tokens to generate
        num_return_sequences: The number of sequences to generate
    
    Returns:
        JSON response with OCR'd text for the specified page(s)
    """
    try:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pdf_path = os.path.join(script_dir, "pdfs", pdf_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error rendering PDF: " + str(e) )
    # Get the number of pages in the PDF
    num_pages = total_pages(pdf_path)

    print(f"Total number of pages of {pdf_path}: {num_pages}")
    

    # Set up the result dictionary
    result_dict = {"pages": []}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = []

    for page in range(3): #num_pages
        output.append(process_page(page_num=page,pdf_path=pdf_path, 
            temperature=temprature, 
            dpi=dpi, 
            max_new_tokens=max_new_tokens, 
            num_return_sequences=num_return_sequences,
            device=device))

    output.sort(key=lambda x: x['page_number'])
    return output


    # # Use ThreadPoolExecutor since we're primarily I/O bound with the model inference
    # with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    #     # Create a partial function with the fixed parameters
    #     process_func = partial(
    #         process_page, 
    #         pdf_path=pdf_path, 
    #         temperature=temprature, 
    #         dpi=dpi, 
    #         max_new_tokens=max_new_tokens, 
    #         num_return_sequences=num_return_sequences,
    #         device=device
    #     )
        
    #     # Process all pages in parallel
    #     for result in executor.map(process_func, range(3)): #num_pages for entire pdf
    #         result_dict["pages"].append(result)
    #         full_text += f"\n\n--- PAGE {result['page']} ---\n\n{result['text']}"

    # Sort the pages by page number
    # result_dict["pages"] = sorted(result_dict["pages"], key=lambda x: x["page"])
    # result_dict["full_text"] = full_text
    
    # return result_dict