import traceback
import click
import os
from pydantic import BaseModel, Field
from starlette.responses import HTMLResponse
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered
import base64
from contextlib import asynccontextmanager
from typing import Optional, Annotated, Dict, Any, List
import io
import json
import asyncio
import threading
from fastapi import APIRouter, FastAPI, Form, File, HTTPException, UploadFile, BackgroundTasks
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.settings import settings
from govdocs_api.supabase.db_functions import supabase, create_ocr_request, update_ocr_request_status, get_document_by_barcode, create_ocr_job
import tempfile
import img2pdf
import shutil

app_data = {}

# Create a dictionary to store active processing threads
active_requests = {}

def run_in_thread(target_function):
    """Decorator to run a function in a separate thread."""
    def run(*args, **kwargs):
        t = threading.Thread(target=target_function, args=args, kwargs=kwargs)
        t.daemon = True  # Daemonize thread to allow program to exit
        t.start()
        return t
    return run

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
    barcode: Annotated[
        Optional[int], Field(description="Barcode for the PDF")
    ]
    filepath: Annotated[
        Optional[str], Field(description="The filepath for the temporary PDF.")
    ] = None
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
    llm_service: Annotated[
        str,
        Field(description="The LLM service to use for parsing ocr")
    ] = "marker.services.claude.ClaudeService"


async def _convert_pdf(params: CommonParams) -> Dict[str, Any]: 
    assert params.output_format in ["markdown", "json", "html"], "Invalid output format"

    claude_api_key=os.getenv("CLAUDE_API_KEY")
    claude_model=os.getenv("CLAUDE_MODEL")
    
    # Parse the page_range string to get all page numbers
    page_numbers = []
    # When constructing page numbers, translate from user-provided numbers to 0-indexed
    if params.page_range:
        parts = params.page_range.split(',')
        for part in parts:
            if '-' in part:
                # Handle range like "1-5"
                start, end = map(int, part.split('-'))
    
                page_numbers.extend(range(start, end + 1))
            else:
                # Handle individual page like "1"
                try:
                    page_numbers.append(int(part)) 
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid page number in page range: {part}")

    # Sort the page numbers
    page_numbers.sort()

    # Create a temporary directory for the downloaded images
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")

    # Disable tokenizers parallelism to avoid warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Download each page and save to the temp directory
    image_paths = []
    for page_num in page_numbers:
        try:
            # Download image from Supabase Storage
            response = (
                supabase.storage
                .from_("ia_bucket")
                .download(f"{params.barcode}/{page_num}.png")
            )

            print(f"Downloaded image for barcode {params.barcode}, page {page_num}")

            # Save the image to a file in the temp directory
            image_path = os.path.join(temp_dir, f"{page_num}.png")
            with open(image_path, "wb") as f:
                f.write(response)
            
            image_paths.append(image_path)
            print(f"Saved image to {image_path}")
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Could not find image for barcode {params.barcode}, page {page_num}: {str(e)}")

    # For PDF converter, we need to convert images to a single PDF
    pdf_path = os.path.join(temp_dir, f"{params.barcode}.pdf")
    try:
        with open(pdf_path, "wb") as f:
            f.write(img2pdf.convert(image_paths))
        params.filepath = pdf_path
        print(f"Created PDF at {pdf_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create PDF: {str(e)}")
    try:
        params.page_range = None
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
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    encoded = {}
    for k, v in images.items():
        byte_stream = io.BytesIO()
        v.save(byte_stream, format=settings.OUTPUT_IMAGE_FORMAT)
        encoded[k] = base64.b64encode(byte_stream.getvalue()).decode(settings.OUTPUT_ENCODING)

    return {
        "success": True,
        "text": text,
        "metadata": metadata,
        "images": encoded,
    }

@run_in_thread
def process_marker_request_thread(request_id: int, document_id: str, barcode: int, page_range: str, 
                                 languages: Optional[str], force_ocr: bool, paginate_output: bool, 
                                 output_format: str, llm_service: str):
    """
    Process Marker OCR in a separate thread and save results to the database.
    This function runs in its own thread to avoid blocking the FastAPI event loop.
    """
    try:
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the async processing function in this thread's event loop
        loop.run_until_complete(
            _process_marker_request(
                request_id, document_id, barcode, page_range,
                languages, force_ocr, paginate_output, output_format, llm_service
            )
        )
    except Exception as e:
        print(f"Thread error processing Marker OCR request {request_id}: {str(e)}")
        traceback.print_exc()
        # We need to run the status update in the thread's event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(update_ocr_request_status(request_id, "error"))
    finally:
        # Clean up
        if request_id in active_requests:
            del active_requests[request_id]

async def _process_marker_request(request_id: int, document_id: str, barcode: int, page_range: str, 
                               languages: Optional[str], force_ocr: bool, paginate_output: bool, 
                               output_format: str, llm_service: str):
    """
    Process Marker OCR in the background and save results to the database.
    This is the actual processing function that runs inside the thread.
    """
    try:
        params = CommonParams(
            barcode=barcode,
            page_range=page_range,
            languages=languages,
            force_ocr=force_ocr,
            paginate_output=paginate_output,
            output_format=output_format,
            llm_service=llm_service
        )
        
        # Perform OCR
        result = await _convert_pdf(params)
        
        if not result["success"]:
            print(f"Error processing Marker OCR: {result.get('error', 'Unknown error')}")
            await update_ocr_request_status(request_id, "error")
            return
        
        # Get the page numbers from the page_range
        page_numbers = []
        parts = page_range.split(',')
        for part in parts:
            if '-' in part:
                # Handle range like "1-5"
                start, end = map(int, part.split('-'))
                page_numbers.extend(range(start, end + 1))
            else:
                # Handle individual page like "1"
                try:
                    page_numbers.append(int(part)) 
                except ValueError:
                    continue
        
        # Sort page numbers
        page_numbers.sort()
        
        # Save OCR text to database
        ocr_config = {
            "languages": languages,
            "force_ocr": force_ocr,
            "paginate_output": paginate_output,
            "output_format": output_format,
            "llm_service": llm_service
        }
        
        # For now, save the entire text as a single job for the first page
        # In the future, this could be improved to split by page if the text contains page markers
        for page_num in page_numbers:
            await create_ocr_job(
                request_id=request_id,
                document_id=document_id,
                page_number=page_num,
                ocr_output=result["text"],
                ocr_model="marker",
                ocr_config=ocr_config
            )
        
        # Update request status to completed
        await update_ocr_request_status(request_id, "completed")
        print(f"Marker OCR request {request_id} completed successfully")
        
    except Exception as e:
        error_msg = f"Error processing Marker OCR request {request_id}: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        # Update request status to error
        await update_ocr_request_status(request_id, "error")

@marker.get("/marker")
async def convert_pdf(
    barcode: int,
    page_range: str,
    languages: Optional[str] = None,
    force_ocr: bool = False,
    paginate_output: bool = False,
    output_format: str = "html"
): 
    """
    Perform OCR on specified pages using Marker.
    
    Args:
        barcode: Barcode identifier for the document
        page_range: Page range to process in format like "1-5,7,9-11"
        languages: Optional languages to use for OCR
        force_ocr: Whether to force OCR on all pages
        paginate_output: Whether to paginate the output
        output_format: The format to output the text in
        
    Returns:
        JSON response with request ID and status
    """
    try:
        # Get document from database
        document = await get_document_by_barcode(str(barcode))
        if not document:
            raise HTTPException(status_code=404, detail=f"Document with barcode {barcode} not found")
        
        document_id = document["id"]
        ocr_config = {
            "languages": languages,
            "force_ocr": force_ocr,
            "paginate_output": paginate_output,
            "output_format": output_format,
            "llm_service": "marker.services.claude.ClaudeService"
        }
        
        # Create request record
        request_record = await create_ocr_request(
            document_id=document_id,
            page_range=page_range,
            ocr_model="marker",
            ocr_config=ocr_config
        )
        
        if not request_record:
            raise HTTPException(status_code=500, detail="Failed to create OCR request")
        
        request_id = request_record["id"]
        
        # Start processing in a separate thread instead of using BackgroundTasks
        thread = process_marker_request_thread(
            request_id=request_id,
            document_id=document_id,
            barcode=barcode,
            page_range=page_range,
            languages=languages,
            force_ocr=force_ocr,
            paginate_output=paginate_output,
            output_format=output_format,
            llm_service="marker.services.claude.ClaudeService"
        )
        
        # Store the thread reference
        active_requests[request_id] = thread
        
        return {
            "message": "Marker OCR processing started",
            "request_id": request_id,
            "status": "processing",
            "document_id": document_id,
            "page_range": page_range
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing OCR: {str(e)}")

@marker.get("/marker/status/{request_id}")
async def marker_status(request_id: int):
    """
    Get the status of a Marker OCR request.
    
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

@marker.get("/marker/result/{request_id}")
async def marker_result(request_id: int):
    """
    Get the results of a completed Marker OCR request.
    
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
        #jobs = supabase.table("ocr_jobs").select("*").eq("request_id", request_id).order("page_number", {"ascending": True}).execute()
        jobs = supabase.table("ocr_jobs").select("*").eq("request_id", request_id).order("page_number", desc=False).execute()
        
        if not jobs.data:
            raise HTTPException(status_code=404, detail=f"No OCR jobs found for request ID {request_id}")
        
        # Since Marker currently saves the full text to each page job,
        # just return the text from the first job
        ocr_text = jobs.data[0]["ocr_output"]
        
        return {
            "text": ocr_text,
            "page_range": request_data["page_range"]
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting OCR results: {str(e)}")


@marker.post("/marker/upload")
async def convert_pdf_upload(
    background_tasks: BackgroundTasks,
    page_range: Optional[str] = Form(default=None),
    languages: Optional[str] = Form(default=None),
    force_ocr: Optional[bool] = Form(default=False),
    paginate_output: Optional[bool] = Form(default=False),
    output_format: Optional[str] = Form(default="markdown"),
    file: UploadFile = File(
        ..., description="The PDF file to convert.", media_type="application/pdf"
    ),
):
    """
    Upload and perform OCR on a PDF file using Marker.
    
    Args:
        page_range: Page range to process
        languages: Optional languages to use for OCR
        force_ocr: Whether to force OCR on all pages
        paginate_output: Whether to paginate the output
        output_format: The format to output the text in
        file: The PDF file to convert
        
    Returns:
        JSON response with OCR results
    """
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
        llm_service="marker.services.claude.ClaudeService"
    )
    
    # For direct uploads, we'll still process synchronously
    # since there's no way to check status later
    results = await _convert_pdf(params)
    os.remove(upload_path)
    return results


