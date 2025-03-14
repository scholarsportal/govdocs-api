from pydantic import BaseModel
from typing import Optional, List

class TesseactOCRRequest(BaseModel):
  pdf_path: str
  config: Optional[str] = ""

class MarkerOCRRequest(BaseModel):
  pdf_path: str
  config: Optional[str] = ""

class GOTOCRRequest(BaseModel):
  pdf_path: str
  temprature: int
  max_new_tokens: Optional[int] = 4096
  do_sample: Optional[bool] = True

class OLMOCRRequest(BaseModel):
  pdf_path: str
  temprature: int
  max_new_tokens: Optional[int] = 4096
  do_sample: Optional[bool] = True
