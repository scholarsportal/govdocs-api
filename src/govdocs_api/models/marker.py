# For marker OCR
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
import os

# Marker OCR implementation
def marker_ocr(pdf_path, output_dir):
    output_file = os.path.join(output_dir, 'marker_output.txt')
    output_md_file = os.path.join(output_dir, 'marker_output.md')
    os.makedirs(output_dir, exist_ok=True)
    
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )
    
    rendered = converter(pdf_path)
    text,_, images = text_from_rendered(rendered)
    
    # Save the markdown text
    with open(output_md_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    # Save a plain text version
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    # Save images
    for key, img in images.items():
        img_path = os.path.join(output_dir, key)
        img.save(img_path)
    
    return text