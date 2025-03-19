# For GOT OCR
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch 
#from govdocs_api.utilities.pdf_utilities import convert_pdf_to_images
import os
import gc


# # GOT OCR implementation
# def got_ocr(pdf_path, output_dir):
#     output_file = os.path.join(output_dir, 'got_ocr_output.txt')
#     os.makedirs(output_dir, exist_ok=True)
    
#     got_ocr_model = AutoModelForImageTextToText.from_pretrained(
#         "stepfun-ai/GOT-OCR-2.0-hf"
#     ).to("cuda")
#     got_ocr_processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")
    
#     # Convert PDF to images
#     #images = convert_pdf_to_images(pdf_path)
    
#     full_text = ""
#     for i, image in enumerate(images):
#         # Save the image temporarily
#         temp_img_path = os.path.join(output_dir, f"temp_page_{i}.jpg")
#         image.save(temp_img_path)
        
#         # Process with GOT OCR
#         inputs = got_ocr_processor(temp_img_path, return_tensors="pt").to("cuda:0")
#         generate_ids = got_ocr_model.generate(
#             **inputs,
#             do_sample=False,
#             tokenizer=got_ocr_processor.tokenizer,
#             stop_strings="<|im_end|>",
#             max_new_tokens=4096,
#         )
#         page_text = got_ocr_processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
#         full_text += f"\n\n--- PAGE {i+1} ---\n\n"
#         full_text += page_text
        
#         # Remove temp image
#         os.remove(temp_img_path)
    
#     # Save the output
#     with open(output_file, 'w', encoding='utf-8') as f:
#         f.write(full_text)
    
#     # Clean up the model to free memory
#     del got_ocr_model
#     del got_ocr_processor
#     torch.cuda.empty_cache()
#     gc.collect()
    
#     return full_text