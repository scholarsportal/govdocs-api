from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from govdocs_api.utilities.pdf_utilities import render_pdf_to_base64png
from olmocr.prompts import build_openai_silver_data_prompt
from olmocr.prompts.anchor import get_anchor_text
import os
import torch
import json
import gc
from io import BytesIO
from PIL import Image
import base64


# OLM OCR implementation
def olm_ocr(pdf_path, output_dir):
    output_file = os.path.join(output_dir, 'olm_ocr_output.txt')
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "allenai/olmOCR-7B-0225-preview", 
        torch_dtype=torch.bfloat16, 
        cache_dir="./models"
    ).eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Get the number of pages in the PDF
    from pypdf import PdfReader
    pdf = PdfReader(pdf_path)
    num_pages = len(pdf.pages)
    
    full_text = ""
    
    for page_num in range(num_pages):
        # Render page to an image
        image_base64 = render_pdf_to_base64png(pdf_path, page_num, target_longest_image_dim=1024)
        
        # Get anchor text
        try:
            anchor_text = get_anchor_text(pdf_path, page_num, pdf_engine="pdfreport", target_length=4000)
            prompt = build_openai_silver_data_prompt(anchor_text)
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
            temperature=0.9,
            max_new_tokens=5000,
            num_return_sequences=1,
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
        
        full_text += f"\n\n--- PAGE {page_num+1} ---\n\n"
        full_text += page_text
    
    # Save the output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    # Clean up the model to free memory
    del model
    del processor
    torch.cuda.empty_cache()
    gc.collect()
    
    return full_text