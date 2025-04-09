from govdocs_api.administrative.admin import admin_router
from govdocs_api.models.smoldocling import smoldocling
from govdocs_api.models.olmocr import olm_ocr
from govdocs_api.models.rolmocr import rolmocr
from govdocs_api.models.marker import marker
from govdocs_api.models.tesseract import tesseract
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import logging

# Configure logging to avoid duplicates
logger = logging.getLogger("uvicorn.error")
logger.propagate = False
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(levelname)s %(asctime)s - %(message)s", datefmt="%m-%d %H:%M:%S"))
    logger.addHandler(handler)

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
app.include_router(rolmocr)
app.include_router(smoldocling)
app.include_router(admin_router)
