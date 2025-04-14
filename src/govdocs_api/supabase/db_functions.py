import os
from supabase import create_client, Client, acreate_client,  AsyncClient
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional
import json

load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_ANON_KEY")
print(f'\nSupabase URL: {url}\nSupabase Key: {key}')
supabase: Client = create_client(url, key)

async def create_supabase() -> AsyncClient:
    return await acreate_client(
        url,
        key,
    )

# OCR database functions
async def create_ocr_request(document_id: str, page_range: str, ocr_model: str, ocr_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new OCR request in the database.
    
    Args:
        document_id: UUID of the document
        page_range: Page range to process
        ocr_model: Name of the OCR model to use
        ocr_config: Configuration parameters for the OCR process
        
    Returns:
        The created record
    """
    response = supabase.table("ocr_requests").insert({
        "document_id": document_id,
        "page_range": page_range,
        "ocr_model": ocr_model,
        "ocr_config": json.dumps(ocr_config),
        "status": "processing"
    }).execute()
    
    return response.data[0] if response.data else None

async def update_ocr_request_status(request_id: int, status: str) -> Dict[str, Any]:
    """
    Update the status of an OCR request.
    
    Args:
        request_id: ID of the OCR request
        status: New status ('processing', 'completed', 'error')
        
    Returns:
        The updated record
    """
    response = supabase.table("ocr_requests").update({
        "status": status
    }).eq("id", request_id).execute()
    
    return response.data[0] if response.data else None

async def get_document_by_barcode(barcode: str) -> Dict[str, Any]:
    """
    Get a document record by its barcode.
    
    Args:
        barcode: The document barcode
        
    Returns:
        The document record or None if not found
    """
    response = supabase.table("documents").select("*").eq("barcode", barcode).execute()
    
    return response.data[0] if response.data and len(response.data) > 0 else None

async def get_document_page_count(barcode: str) -> int:
    """
    Get the total number of pages for a document by checking pages stored in Supabase.
    
    Args:
        barcode: The document barcode
        
    Returns:
        The total number of pages found, or 0 if document not found
    """
    try:
        # List files in the document's folder to count pages
        response = supabase.storage.from_("ia_bucket").list(barcode)
        
        # Count the PNG files (typically named 1.png, 2.png, etc.)
        page_count = sum(1 for item in response if item['name'].lower().endswith('.png'))
        
        return page_count
    except Exception as e:
        print(f"Error getting page count for document {barcode}: {str(e)}")
        return 0

async def create_ocr_job(request_id: int, document_id: str, page_number: int, 
                         ocr_output: str, ocr_model: str, ocr_config: Dict[str, Any], 
                         status: str = "completed") -> Dict[str, Any]:
    """
    Create a new OCR job record.
    
    Args:
        request_id: ID of the parent OCR request
        document_id: UUID of the document
        page_number: Page number processed
        ocr_output: The OCR output text/content
        ocr_model: Name of the OCR model used
        ocr_config: Configuration parameters used for the OCR process
        status: Job status
        
    Returns:
        The created record
    """
    response = supabase.table("ocr_jobs").insert({
        "request_id": request_id,
        "document_id": document_id,
        "page_number": page_number,
        "ocr_output": ocr_output,
        "ocr_model": ocr_model,
        "ocr_config": json.dumps(ocr_config),
        "status": status
    }).execute()
    
    return response.data[0] if response.data else None

async def get_ocr_job(document_id: str, page_number: int, ocr_model: str, ocr_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get an OCR job by document ID, page number, model, and config.
    
    Args:
        document_id: UUID of the document
        page_number: Page number processed
        ocr_model: Name of the OCR model used
        ocr_config: Configuration parameters used for the OCR process
        
    Returns:
        The OCR job record or None if not found
    """
    response = supabase.table("ocr_jobs").select("*") \
        .eq("document_id", document_id) \
        .eq("page_number", page_number) \
        .eq("ocr_model", ocr_model) \
        .eq("ocr_config", json.dumps(ocr_config)) \
        .execute()
    
    return response.data[0] if response.data and len(response.data) > 0 else None