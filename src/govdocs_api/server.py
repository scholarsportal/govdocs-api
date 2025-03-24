
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os

## IGNORE
os.environ["TRANSFORMERS_CACHE"] = "/local/home/hfurquan/myProjects/Leaderboard/cache"
os.environ["HF_HOME"] = "/local/home/hfurquan/myProjects/Leaderboard/cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
##

from govdocs_api.models.tesseract import tesseract
from govdocs_api.models.marker import marker
from govdocs_api.models.olmocr import olm_ocr
from govdocs_api.models.smoldocling import smoldocling
from govdocs_api.administrative.admin import admin_router

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

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")