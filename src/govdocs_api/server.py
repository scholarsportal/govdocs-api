from govdocs_api.administrative.admin import admin_router
from govdocs_api.models.smoldocling import smoldocling
from govdocs_api.models.olmocr import olm_ocr
from govdocs_api.models.marker import marker
from govdocs_api.models.tesseract import tesseract
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import logging
import sys

# Configure logging for the entire application
logger = logging.getLogger("uvicorn.error")
logger.propagate = False
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    # Add console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        "%(levelname)s %(asctime)s - %(message)s", datefmt="%m-%d %H:%M:%S"))
    logger.addHandler(console_handler)
    
    # Add file handler for persistent logging
    try:
        file_handler = logging.FileHandler("govdocs_api.log")
        file_handler.setFormatter(logging.Formatter(
            "%(levelname)s %(asctime)s [%(name)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not set up file logging: {str(e)}")

# Create a logger specific to background tasks
background_logger = logging.getLogger("background_tasks")
background_logger.setLevel(logging.INFO)
for handler in logger.handlers:
    background_logger.addHandler(handler)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tesseract)
app.include_router(marker)
app.include_router(olm_ocr)
app.include_router(smoldocling)
app.include_router(admin_router)

# Make the loggers available to other modules
app.state.logger = logger
app.state.background_logger = background_logger
