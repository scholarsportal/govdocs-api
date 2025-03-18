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


import traceback

import click
import os

from pydantic import BaseModel, Field
from starlette.responses import HTMLResponse

from marker.config.parser import ConfigParser
from marker.output import text_from_rendered

import base64
from contextlib import asynccontextmanager
from typing import Optional, Annotated
import io

from fastapi import APIRouter, FastAPI, Form, File, HTTPException, UploadFile
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.settings import settings

app_data = {}


UPLOAD_DIRECTORY = "./uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
app_data = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    app_data["models"] = create_model_dict()

    yield

    if "models" in app_data:
        del app_data["models"]

marker = APIRouter(lifespan=lifespan)


@marker.get("/")
async def root():
    return HTMLResponse(
        """
<h1>Marker API</h1>
<ul>
    <li><a href="/docs">API Documentation</a></li>
    <li><a href="/marker">Run marker (post request only)</a></li>
</ul>
"""
    )


class CommonParams(BaseModel):
    filepath: Annotated[
        Optional[str], Field(description="The path to the PDF file to convert.")
    ]
    page_range: Annotated[
        Optional[str],
        Field(description="Page range to convert, specify comma separated page numbers or ranges.  Example: 0,5-10,20", example=None)
    ] = None
    languages: Annotated[
        Optional[str],
        Field(description="Comma separated list of languages to use for OCR. Must be either the names or codes from from https://github.com/VikParuchuri/surya/blob/master/surya/recognition/languages.py.", example=None)
    ] = None
    force_ocr: Annotated[
        bool,
        Field(
            description="Force OCR on all pages of the PDF.  Defaults to False.  This can lead to worse results if you have good text in your PDFs (which is true in most cases)."
        ),
    ] = False
    paginate_output: Annotated[
        bool,
        Field(
            description="Whether to paginate the output.  Defaults to False.  If set to True, each page of the output will be separated by a horizontal rule that contains the page number (2 newlines, {PAGE_NUMBER}, 48 - characters, 2 newlines)."
        ),
    ] = False
    output_format: Annotated[
        str,
        Field(description="The format to output the text in.  Can be 'markdown', 'json', or 'html'.  Defaults to 'markdown'.")
    ] = "markdown"


async def _convert_pdf(params: CommonParams) -> HTMLResponse: 
    assert params.output_format in ["markdown", "json", "html"], "Invalid output format"
    try:
        options = params.model_dump()
        print(options)
        config_parser = ConfigParser(options)
        config_dict = config_parser.generate_config_dict()
        config_dict["pdftext_workers"] = 1
        converter_cls = PdfConverter
        converter = converter_cls(
            config=config_dict,
            artifact_dict=app_data["models"],
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service()
        )
        rendered = converter(params.filepath)
        text, _, images = text_from_rendered(rendered)
        metadata = rendered.metadata
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
        }

    encoded = {}
    for k, v in images.items():
        byte_stream = io.BytesIO()
        v.save(byte_stream, format=settings.OUTPUT_IMAGE_FORMAT)
        encoded[k] = base64.b64encode(byte_stream.getvalue()).decode(settings.OUTPUT_ENCODING)

    # return {
    #     "format": params.output_format,
    #     "output": text,
    #     "images": encoded,
    #     "metadata": metadata,
    #     "success": True,
    # }
    return text

@marker.get("/marker")
async def convert_pdf(
    filepath: str,
    page_range: str,
    languages: Optional[str] = None,
    force_ocr: bool = False,
    paginate_output: bool = False,
    output_format: str = "html"
): 
    try:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pdf_path = os.path.join(script_dir, "pdfs", filepath)
        #images = [render_pdf_to_base64png(local_pdf_path=pdf_path, page_num=page_number, target_longest_image_dim=1024)]
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error rendering PDF: " + str(e) )

    params = CommonParams(
        filepath=pdf_path,
        page_range=page_range,
        languages=languages,
        force_ocr=force_ocr,
        paginate_output=paginate_output,
        output_format=output_format
    )
    return await _convert_pdf(params)



@marker.post("/marker/upload")
async def convert_pdf_upload(
    page_range: Optional[str] = Form(default=None),
    languages: Optional[str] = Form(default=None),
    force_ocr: Optional[bool] = Form(default=False),
    paginate_output: Optional[bool] = Form(default=False),
    output_format: Optional[str] = Form(default="markdown"),
    file: UploadFile = File(
        ..., description="The PDF file to convert.", media_type="application/pdf"
    ),
):
    upload_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(upload_path, "wb+") as upload_file:
        file_contents = await file.read()
        upload_file.write(file_contents)

    params = CommonParams(
        filepath=upload_path,
        page_range=page_range,
        languages=languages,
        force_ocr=force_ocr,
        paginate_output=paginate_output,
        output_format=output_format,
    )
    results = await _convert_pdf(params)
    os.remove(upload_path)
    return results


