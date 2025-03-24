from fastapi import FastAPI, HTTPException, Request, Response, status, Depends, BackgroundTasks, APIRouter
from fastapi.responses import JSONResponse
import httpx
import os
import uuid
import tempfile
import shutil
from typing import List, Dict, Optional, Union
import time
from datetime import datetime
import asyncio
import base64
from io import BytesIO
from pathlib import Path
import logging
from supabase import create_client, Client
from pydantic import BaseModel, Field
import concurrent.futures
from govdocs_api.utilities.pdf_utilities import render_pdf_to_base64png, total_pages
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_ANON_KEY")
print(f'Supbase Url {supabase_url} Supabase Key {supabase_key}')
supabase: Client = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None


# Create a temporary directory for downloaded files
TEMP_DIR = Path(os.path.join(tempfile.gettempdir(), "govdocs-downloads"))
TEMP_DIR.mkdir(exist_ok=True)

# Models for request/response
class DocumentRequest(BaseModel):
    ia_link: str = Field(..., description="Internet Archive link to the document")
    barcode: int = Field(..., description="Document barcode")
    title: str = Field(..., description="Document title")
    max_pages: int = Field(100, description="Maximum number of pages to process")

class ProcessingStatus(BaseModel):
    document_id: str
    status: str
    pages_processed: int
    total_pages: Optional[int] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

admin_router = APIRouter(prefix="/admin", tags=["admin"])

# Utility functions for document processing
def get_ia_download_url(barcode: Union[int, str]) -> str:
    """Generate the direct download URL for an Internet Archive item."""
    return f"https://archive.org/download/{barcode}/{barcode}.pdf"

async def download_file(url: str, dest_path: Path) -> Path:
    """Download a file from a URL to the specified path."""
    logger.info(f"Downloading {url} to {dest_path}")

    timeout = httpx.Timeout(60.0, connect=10.0)  # 60s read timeout, 10s connect timeout
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url, follow_redirects=True)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, 
                               detail=f"Failed to download file: {response.text}")
        
        with open(dest_path, 'wb') as f:
            f.write(response.content)
    
    logger.info(f"Download completed: {dest_path}")
    return dest_path

async def upload_to_supabase(barcode: str, page_num: int, image_base64: str) -> dict:
    """Upload a rendered page image to Supabase storage."""
    try:
        # Decode base64 to binary
        image_data = base64.b64decode(image_base64)
        
        # Construct file path in the bucket
        file_path = f"{barcode}/{page_num}.png"
        
        # Upload file
        response = supabase.storage \
            .from_("ia_bucket") \
            .upload(file_path, image_data, {"content-type": "image/png"})
        
        # Get public URL for the uploaded file
        public_url = supabase.storage \
            .from_("ia_bucket") \
            .get_public_url(file_path)
        
        return {"path": file_path, "url": public_url}
    
    except Exception as e:
        logger.error(f"Error uploading to Supabase: {str(e)}")
        raise

async def update_processing_status(document_id: str, status: str, pages_processed: int = None, 
                                  total_pages: int = None, error_message: str = None) -> None:
    """Update the processing status of a document in the database."""
    update_data = {"status": status, "updated_at": "now()"}
    
    if pages_processed is not None:
        update_data["pages_processed"] = pages_processed
    
    if total_pages is not None:
        update_data["total_pages"] = total_pages
    
    if error_message is not None:
        update_data["error_message"] = error_message
    
    try:
        supabase.table("document_processing") \
            .update(update_data) \
            .eq("document_id", document_id) \
            .execute()
    except Exception as e:
        logger.error(f"Error updating processing status: {str(e)}")

async def process_document(document_id: str, ia_link: str, barcode: str, max_pages: int = 100) -> None:
    """
    Process a document by downloading it from Internet Archive, 
    rendering pages, and uploading them to Supabase storage.
    """
    # Update status to processing
    await update_processing_status(document_id, "processing")
    
    pdf_path = None
    try:
        # Download PDF
        download_url = get_ia_download_url(barcode)
        pdf_path = TEMP_DIR / f"{barcode}.pdf"
        await download_file(download_url, pdf_path)

        print(f"PDF download complete ✅")
        
        # Get total number of pages
        num_pages = total_pages(pdf_path)
        pages_to_process = min(num_pages, max_pages)
        
        await update_processing_status(document_id, "processing", 0, pages_to_process)

        print(f"PDF processing status updated ✅")
        
        # Process pages in batches
        batch_size = 10
        for batch_start in range(1, pages_to_process + 1, batch_size):
            batch_end = min(batch_start + batch_size - 1, pages_to_process)
            tasks = []
            
            for page_num in range(batch_start, batch_end + 1):
                tasks.append(process_page(document_id, str(barcode), pdf_path, page_num))
            
            # Process batch concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check for exceptions and update progress
            successful_pages = sum(1 for r in results if not isinstance(r, Exception))
            await update_processing_status(document_id, "processing", batch_end)
            
        # Mark as completed
        await update_processing_status(document_id, "completed", pages_to_process, pages_to_process)
        
    except Exception as e:
        logger.exception(f"Error processing document {barcode}: {str(e)}")
        await update_processing_status(
            document_id, "error", error_message=str(e)
        )
    finally:
        # Clean up downloaded file
        if pdf_path and pdf_path.exists():
            pdf_path.unlink()

async def process_page(document_id: str, barcode: str, pdf_path: Path, page_num: int) -> dict:
    """Process a single page from the PDF."""
    try:
        # Render page to image
        logger.info(f"Rendering page {page_num} of document {barcode}")
        image_base64 = render_pdf_to_base64png(str(pdf_path), page_num, target_longest_image_dim=1024)
        
        # Upload to Supabase
        result = await upload_to_supabase(barcode, page_num, image_base64)
        
        logger.info(f"Processed page {page_num} of document {barcode}")
        return result
    except Exception as e:
        logger.error(f"Error processing page {page_num} of document {barcode}: {str(e)}")
        raise

# Endpoints
@admin_router.post("/process-document", response_model=ProcessingStatus)
async def start_document_processing(document: DocumentRequest, background_tasks: BackgroundTasks):
    """
    Start the document processing pipeline:
    1. Download the PDF from Internet Archive
    2. Render pages as PNG images
    3. Upload rendered images to Supabase storage
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client is not configured")
    
    try:
        # Insert document into documents table first
        document_insert = supabase.table("documents").insert({
            "title": document.title,
            "ia_link": document.ia_link,
            "barcode": document.barcode
        }).execute()
        
        document_id = document_insert.data[0]['id']
        
        # Create a processing entry
        processing_insert = supabase.table("document_processing").insert({
            "document_id": document_id,
            "status": "pending",
            "pages_processed": 0
        }).execute()
        
        # Start processing in the background
        background_tasks.add_task(
            process_document, 
            document_id, 
            document.ia_link, 
            str(document.barcode),
            document.max_pages
        )
        
        return {
            "document_id": document_id,
            "status": "pending",
            "pages_processed": 0,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
    
    except Exception as e:
        logger.exception("Error starting document processing")
        raise HTTPException(status_code=500, detail=str(e))

@admin_router.get("/document-status/{document_id}", response_model=ProcessingStatus)
async def get_document_status(document_id: str):
    """Get the current processing status of a document."""
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client is not configured")
    
    try:
        result = supabase.table("document_processing") \
            .select("*") \
            .eq("document_id", document_id) \
            .execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Document processing record not found")
        
        return result.data[0]
    
    except Exception as e:
        logger.exception(f"Error getting document status for {document_id}")
        raise HTTPException(status_code=500, detail=str(e))

@admin_router.get("/documents", response_model=List[Dict])
async def list_documents():
    """List all documents with their processing status."""
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client is not configured")
    
    try:
        result = supabase.from_("documents") \
            .select("*") \
            .execute()
        
        return result.data
    
    except Exception as e:
        logger.exception("Error listing documents")
        raise HTTPException(status_code=500, detail=str(e))

# Add cleanup endpoint to clear temporary files (for admin use)
@admin_router.post("/cleanup-temp")
async def cleanup_temp_files():
    """Clean up temporary downloaded files."""
    try:
        for file in TEMP_DIR.glob("*.pdf"):
            file.unlink()
        return {"message": "Temporary files cleaned up successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning up files: {str(e)}")

