from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from govdocs_api.utilities.pdf_utilities import render_pdf_to_base64png, total_pages
import os
import torch
import json
import gc
from io import BytesIO
from PIL import Image
import base64
import hashlib
import atexit
from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
from functools import partial, cache
import time
import httpx
import asyncio
from pypdf import PdfReader
from tqdm import tqdm
from dataclasses import dataclass
from urllib.parse import urlparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from typing import List, Dict, Any, Optional, Tuple
import multiprocessing
import logging
import re

from olmocr.check import (
    check_poppler_version,
    check_sglang_version,
    check_torch_gpu_available,
)
from olmocr.filter.filter import Language, PdfFilter
from olmocr.metrics import MetricsKeeper, WorkerTracker
from olmocr.prompts import PageResponse, build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text
from olmocr.s3_utils import (
    download_zstd_csv,
    expand_s3_glob,
    get_s3_bytes,
    get_s3_bytes_with_backoff,
    parse_s3_path,
)
from olmocr.version import VERSION
from olmocr.work_queue import LocalWorkQueue, S3WorkQueue, WorkQueue


model = None
processor = None
# metrics = MetricsKeeper(window=300)  # 5 minute window
# tracker = WorkerTracker()

# # Process pool for CPU-bound operations
# process_pool = ProcessPoolExecutor(
#     max_workers=min(multiprocessing.cpu_count() // 2 + 1, 16),
#     mp_context=multiprocessing.get_context("spawn")
# )

# # Constants
# SGLANG_SERVER_PORT = 30024  # Adjust as needed

# sglang_logger = logging.getLogger("sglang")
# sglang_logger.setLevel(logging.INFO)
# # Initialize logger
# logger = logging.getLogger("olm-ocr-api")
# logger.setLevel(logging.DEBUG)

# file_handler = logging.FileHandler("olm-ocr-api-debug.log", mode="a")
# file_handler.setLevel(logging.DEBUG)
# file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
# console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# logger.addHandler(file_handler)
# logger.addHandler(console_handler)

# # Metrics tracking
# class MetricsKeeper:
#     def __init__(self, window=300):
#         self.window = window
#         self.metrics = {}
#         self.timestamps = []
    
#     def add_metrics(self, **kwargs):
#         timestamp = time.time()
#         self.timestamps.append(timestamp)
#         for key, value in kwargs.items():
#             if key not in self.metrics:
#                 self.metrics[key] = []
#             self.metrics[key].append(value)
        
#         # Clean old metrics
#         self._clean_old_metrics()
    
#     def _clean_old_metrics(self):
#         if not self.timestamps:
#             return
        
#         cutoff = time.time() - self.window
#         while self.timestamps and self.timestamps[0] < cutoff:
#             self.timestamps.pop(0)
#             for key in self.metrics:
#                 if self.metrics[key]:
#                     self.metrics[key].pop(0)
    
#     def get_summary(self):
#         self._clean_old_metrics()
#         summary = {}
#         for key, values in self.metrics.items():
#             if values:
#                 summary[key] = {
#                     "total": sum(values),
#                     "count": len(values),
#                     "avg": sum(values) / len(values),
#                     "max": max(values),
#                     "min": min(values)
#                 }
#         return summary

# class WorkerTracker:
#     def __init__(self):
#         self.status = {}
#         self.started_at = {}
#         self._lock = asyncio.Lock()
    
#     async def track_work(self, worker_id, work_id, status):
#         async with self._lock:
#             self.status[work_id] = status
#             if status == "started":
#                 self.started_at[work_id] = time.time()
    
#     def get_status(self):
#         active_workers = {}
#         for work_id, status in self.status.items():
#             if status == "started":
#                 elapsed = time.time() - self.started_at.get(work_id, time.time())
#                 active_workers[work_id] = {
#                     "status": status,
#                     "elapsed_seconds": elapsed
#                 }
#         return active_workers

# @dataclass(frozen=True)
# class PageResult:
#     s3_path: str
#     page_num: int
#     response: PageResponse

#     input_tokens: int
#     output_tokens: int
#     is_fallback: bool

# async def sglang_server_task(semaphore):
#     #model_name_or_path = args.model
#     #model_name_or_path = '/local/home/hfurquan/myProjects/Leaderboard/cache/models--allenai--olmOCR-7B-0225-preview'
#     model_name_or_path = "allenai/olmOCR-7B-0225-preview"

#     # Check GPU memory, lower mem devices need a bit less KV cache space because the VLM takes additional memory
#     gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
#     mem_fraction_arg = ["--mem-fraction-static", "0.80"] if gpu_memory < 60 else []

#     cmd = [
#         "python3",
#         "-m",
#         "sglang.launch_server",
#         "--model-path",
#         model_name_or_path,
#         "--chat-template",
#         "qwen2-vl",
#         "--port",
#         str(SGLANG_SERVER_PORT),
#         "--log-level-http",
#         "warning",
#         "--device",
#         "cuda",
#         "--base-gpu-id",
#         "0"
#     ]
#     cmd.extend(mem_fraction_arg)

#     proc = await asyncio.create_subprocess_exec(
#         *cmd,
#         stdout=asyncio.subprocess.PIPE,
#         stderr=asyncio.subprocess.PIPE,
#     )

#     # Ensure the subprocess is terminated on exit
#     def _kill_proc():
#         proc.terminate()

#     atexit.register(_kill_proc)

#     # Shared variables between tasks
#     last_running_req, last_queue_req = 0, 0
#     server_printed_ready_message = False
#     last_semaphore_release = time.time()

#     async def process_line(line):
#         nonlocal last_running_req, last_queue_req, last_semaphore_release, server_printed_ready_message
#         sglang_logger.info(line)

#         # if the server hasn't initialized yet, log all the lines to the main logger also, so that the user
#         # can see any warnings/errors more easily
#         if not server_printed_ready_message:
#             logger.info(line)

#         if "Detected errors during sampling" in line:
#             logger.error("Cannot continue, sampling errors detected, model is probably corrupt")
#             sys.exit(1)

#         # TODO, need to trace down this issue in sglang itself, but it will otherwise cause the server to lock up
#         if "IndexError: list index out of range" in line:
#             logger.error("IndexError in model, restarting server")
#             proc.terminate()

#         if not server_printed_ready_message and "The server is fired up and ready to roll!" in line:
#             server_printed_ready_message = True
#             last_semaphore_release = time.time()

#         match = re.search(r"#running-req: (\d+)", line)
#         if match:
#             last_running_req = int(match.group(1))

#         match = re.search(r"#queue-req: (\d+)", line)
#         if match:
#             last_queue_req = int(match.group(1))
#             logger.info(f"sglang running req: {last_running_req} queue req: {last_queue_req}")

#     async def read_stream(stream):
#         while True:
#             line = await stream.readline()
#             if not line:
#                 break
#             try:
#                 line = line.decode("utf-8").rstrip()
#                 await process_line(line)
#             except Exception as ex:
#                 logger.warning(f"Got {ex} when reading log line from inference server, skipping")

#     async def timeout_task():
#         nonlocal last_running_req, last_queue_req, last_semaphore_release
#         try:
#             while True:
#                 await asyncio.sleep(1)
#                 if server_printed_ready_message and last_queue_req == 0 and time.time() - last_semaphore_release > 30 and semaphore.locked():
#                     semaphore.release()
#                     last_semaphore_release = time.time()
#                     logger.info("Semaphore released, allowing a worker to proceed.")
#         except asyncio.CancelledError:
#             pass  # Clean up if the task is cancelled

#     # Start tasks to read stdout, stderr, and handle timeout logic
#     stdout_task = asyncio.create_task(read_stream(proc.stdout))
#     stderr_task = asyncio.create_task(read_stream(proc.stderr))
#     timeout_task = asyncio.create_task(timeout_task())

#     try:
#         await proc.wait()
#     except asyncio.CancelledError:
#         logger.info("Got cancellation request for SGLang server")
#         proc.terminate()
#         raise

#     timeout_task.cancel()
#     await asyncio.gather(stdout_task, stderr_task, timeout_task, return_exceptions=True)


# async def sglang_server_ready():
#     max_attempts = 300
#     delay_sec = 1
#     url = f"http://localhost:{SGLANG_SERVER_PORT}/v1/models"

#     for attempt in range(1, max_attempts + 1):
#         try:
#             async with httpx.AsyncClient() as session:
#                 response = await session.get(url)

#                 if response.status_code == 200:
#                     logger.info("sglang server is ready.")
#                     return
#                 else:
#                     logger.info(f"Attempt {attempt}: Unexpected status code {response.status_code}")
#         except Exception:
#             logger.warning(f"Attempt {attempt}: Please wait for sglang server to become ready...")

#         await asyncio.sleep(delay_sec)

#     raise Exception("sglang server did not become ready after waiting.")


# # Simple HTTP client for SGLang server communication
# async def apost(url, json_data):
#     parsed_url = urlparse(url)
#     host = parsed_url.hostname
#     port = parsed_url.port or 80
#     path = parsed_url.path or "/"
    
#     request_start = time.perf_counter()
#     writer = None
#     try:
#         reader, writer = await asyncio.open_connection(host, port)
        
#         json_payload = json.dumps(json_data)
#         request = (
#             f"POST {path} HTTP/1.1\r\n"
#             f"Host: {host}\r\n"
#             f"Content-Type: application/json\r\n"
#             f"Content-Length: {len(json_payload)}\r\n"
#             f"Connection: close\r\n\r\n"
#             f"{json_payload}"
#         )
#         writer.write(request.encode())
#         await writer.drain()
        
#         # Read status line
#         status_line = await reader.readline()
#         if not status_line:
#             raise ConnectionError("No response from server")
        
#         status_parts = status_line.decode().strip().split(" ", 2)
#         if len(status_parts) < 2:
#             raise ValueError(f"Malformed status line: {status_line.decode().strip()}")
        
#         status_code = int(status_parts[1])
        
#         # Read headers
#         headers = {}
#         while True:
#             line = await reader.readline()
#             if line in (b"\r\n", b"\n", b""):
#                 break
#             key, _, value = line.decode().partition(":")
#             headers[key.strip().lower()] = value.strip()
        
#         # Read response body
#         if "content-length" in headers:
#             body_length = int(headers["content-length"])
#             response_body = await reader.readexactly(body_length)
#         else:
#             chunks = []
#             while True:
#                 chunk = await reader.read(4096)
#                 if not chunk:
#                     break
#                 chunks.append(chunk)
#             response_body = b''.join(chunks)
        
#         request_end = time.perf_counter()
#         logger.debug(f"SGLang request completed in {request_end - request_start:.2f}s")
        
#         return status_code, response_body
#     except Exception as e:
#         request_end = time.perf_counter()
#         logger.warning(f"SGLang request failed in {request_end - request_start:.2f}s: {str(e)}")
#         raise
#     finally:
#         if writer is not None:
#             try:
#                 writer.close()
#                 await writer.wait_closed()
#             except:
#                 pass

# async def build_page_query(local_pdf_path: str, page: int, target_longest_image_dim: int, 
#                           target_anchor_text_len: int, image_rotation: int = 0) -> dict:
#     query_start = time.perf_counter()
#     MAX_TOKENS = 3000
#     assert image_rotation in [0, 90, 180, 270], "Invalid image rotation provided"
    
#     # Run these operations concurrently
#     render_start = time.perf_counter()
#     image_base64_task = asyncio.create_task(
#         asyncio.to_thread(render_pdf_to_base64png, local_pdf_path, page, 
#                          target_longest_image_dim=target_longest_image_dim)
#     )
    
#     # Get anchor text (CPU-bound operation)
#     anchor_start = time.perf_counter()
#     loop = asyncio.get_running_loop()
#     anchor_text_task = asyncio.create_task(
#         loop.run_in_executor(
#             process_pool, 
#             partial(get_anchor_text, pdf_engine="pdfreport", target_length=target_anchor_text_len), 
#             local_pdf_path, page
#         )
#     )
    
#     # Wait for both operations to complete
#     image_base64, anchor_text = await asyncio.gather(image_base64_task, anchor_text_task)
#     render_end = time.perf_counter()
#     anchor_end = time.perf_counter()
    
#     # Handle image rotation if needed
#     if image_rotation != 0:
#         rotation_start = time.perf_counter()
#         image_bytes = base64.b64decode(image_base64)
#         with Image.open(BytesIO(image_bytes)) as img:
#             rotated_img = img.rotate(-image_rotation, expand=True)
#             buffered = BytesIO()
#             rotated_img.save(buffered, format="PNG")
        
#         image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
#         rotation_end = time.perf_counter()
    
#     # Build the query
#     prompt = build_finetuning_prompt(anchor_text)
#     query = {
#         "model": "Qwen/Qwen2-VL-7B-Instruct",
#         "messages": [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": prompt},
#                     {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
#                 ],
#             }
#         ],
#         "max_tokens": MAX_TOKENS,
#         "temperature": 0.8, 
#     }
    
#     query_end = time.perf_counter()
    
#     performance = {
#         "render_time": render_end - render_start,
#         "anchor_time": anchor_end - anchor_start,
#         "rotation_time": (rotation_end - rotation_start) if image_rotation != 0 else 0,
#         "total_prep_time": query_end - query_start
#     }
    
#     return query, performance

# async def process_page(worker_id: int, pdf_path: str, page_num: int, 
#                      target_longest_image_dim: int = 256, 
#                      target_anchor_text_len: int = 4000,
#                      max_retries: int = 5) -> PageResult:
#     """Process a single page using SGLang with advanced error handling and retries."""
#     COMPLETION_URL = f"http://localhost:{SGLANG_SERVER_PORT}/v1/chat/completions"
#     TEMPERATURE_BY_ATTEMPT = [0.8, 0.9]
    
#     exponential_backoffs = 0
#     local_anchor_text_len = target_anchor_text_len
#     local_image_rotation = 0
#     attempt = 0
#     page_start = time.perf_counter()
    
#     await tracker.track_work(worker_id, f"{pdf_path}-{page_num}", "started")
    
#     performance_metrics = {
#         "attempts": 0,
#         "backoffs": 0
#     }
    
#     while attempt < max_retries:
#         attempt_start = time.perf_counter()
        
#         try:
#             # Build the query for this page
#             query, prep_performance = await build_page_query(
#                 pdf_path, page_num, 
#                 target_longest_image_dim, 
#                 local_anchor_text_len,
#                 image_rotation=local_image_rotation
#             )
            
            
#             performance_metrics.update(prep_performance)
#             logger.info(f"Built page query for {pdf_path} page {page_num+1}, attempt {attempt+1}")
            
#             # Send request to SGLang server
#             sgapi_start = time.perf_counter()
#             status_code, response_body = await apost(COMPLETION_URL, json_data=query)
#             sgapi_end = time.perf_counter()
#             performance_metrics["sglang_time"] = sgapi_end - sgapi_start
            
#             # Handle HTTP errors
#             if status_code == 400:
#                 raise ValueError(f"BadRequestError from server: {response_body}")
#             elif status_code == 500:
#                 raise ValueError(f"InternalServerError from server: {response_body}")
#             elif status_code != 200:
#                 raise ValueError(f"Error http status {status_code}")
            
#             # Parse the response
#             base_response_data = json.loads(response_body)
            
#             # Check if we exceeded token limits
#             if base_response_data["usage"]["total_tokens"] > 8000:  # Adjust as needed
#                 local_anchor_text_len = max(1, local_anchor_text_len // 2)
#                 logger.info(f"Reducing anchor text len to {local_anchor_text_len} for {pdf_path}-{page_num}")
#                 raise ValueError("Response exceeded model_max_context, cannot use this response")
            
#             # Track metrics
#             metrics.add_metrics(
#                 sglang_input_tokens=base_response_data["usage"].get("prompt_tokens", 0),
#                 sglang_output_tokens=base_response_data["usage"].get("completion_tokens", 0),
#                 sglang_latency=(sgapi_end - sgapi_start) * 1000,  # ms
#             )
            
#             # Extract and parse model response
#             model_response_json = json.loads(base_response_data["choices"][0]["message"]["content"])
            
#             # Check if page rotation is needed
#             if model_response_json.get("is_rotation_valid") is False and attempt < max_retries - 1:
#                 rotation_correction = model_response_json.get("rotation_correction", 0)
#                 logger.info(f"Invalid page rotation for {pdf_path}-{page_num}, attempt {attempt+1}. "
#                            f"Retrying with {rotation_correction} rotation.")
#                 local_image_rotation = rotation_correction
#                 raise ValueError(f"Invalid page rotation for {pdf_path}-{page_num}")
            
#             # Success - mark task as complete
#             await tracker.track_work(worker_id, f"{pdf_path}-{page_num}", "finished")
            
#             page_end = time.perf_counter()
#             performance_metrics.update({
#                 "total_time": page_end - page_start,
#                 "attempts": attempt + 1
#             })
            
#             return PageResult(
#                 page_number=page_num + 1,  # Convert to 1-indexed for user
#                 text=model_response_json.get("natural_text", ""),
#                 input_tokens=base_response_data["usage"].get("prompt_tokens", 0),
#                 output_tokens=base_response_data["usage"].get("completion_tokens", 0),
#                 is_fallback=False,
#                 performance=performance_metrics,
#                 metadata={
#                     "primary_language": model_response_json.get("primary_language", "eng"),
#                     "is_table": model_response_json.get("is_table", False),
#                     "is_diagram": model_response_json.get("is_diagram", False),
#                     "rotation": model_response_json.get("rotation_correction", 0)
#                 }
#             )
            
#         except (ConnectionError, OSError, asyncio.TimeoutError) as e:
#             # Network-related errors - use exponential backoff
#             logger.warning(f"Network error on attempt {attempt+1} for {pdf_path}-{page_num}: {type(e).__name__} - {e}")
            
#             sleep_delay = 10 * (2 ** exponential_backoffs)
#             exponential_backoffs += 1
#             performance_metrics["backoffs"] = exponential_backoffs
            
#             logger.info(f"Sleeping for {sleep_delay}s on {pdf_path}-{page_num} to allow server recovery")
#             await asyncio.sleep(sleep_delay)
            
#         except asyncio.CancelledError:
#             # Task was cancelled - propagate the cancellation
#             logger.info(f"Process page {pdf_path}-{page_num} cancelled")
#             await tracker.track_work(worker_id, f"{pdf_path}-{page_num}", "cancelled")
#             raise
            
#         except json.JSONDecodeError as e:
#             # JSON parsing error - retry with next temperature
#             logger.warning(f"JSON decode error on attempt {attempt+1} for {pdf_path}-{page_num}: {e}")
#             attempt += 1
            
#         except ValueError as e:
#             # Value error (includes rotation issues) - retry
#             logger.warning(f"ValueError on attempt {attempt+1} for {pdf_path}-{page_num}: {type(e).__name__} - {e}")
#             attempt += 1
            
#         except Exception as e:
#             # Unexpected error - log and retry
#             logger.exception(f"Unexpected error on attempt {attempt+1} for {pdf_path}-{page_num}: {type(e).__name__} - {e}")
#             attempt += 1
        
#         attempt_end = time.perf_counter()
#         logger.debug(f"Attempt {attempt} for {pdf_path}-{page_num} took {attempt_end - attempt_start:.2f}s")
    
#     # All retries failed - fall back to direct model processing
#     logger.error(f"Failed to process {pdf_path}-{page_num} after {max_retries} attempts, using fallback.")
#     await tracker.track_work(worker_id, f"{pdf_path}-{page_num}", "errored")
    
#     # Use direct model if SGLang failed
#     fallback_start = time.perf_counter()
#     try:
#         # Get anchor text as fallback text
#         anchor_text = await asyncio.to_thread(
#             get_anchor_text, pdf_path, page_num, pdf_engine="pdftotext"
#         )
        
#         fallback_end = time.perf_counter()
#         performance_metrics["fallback_time"] = fallback_end - fallback_start
        
#         return PageResult(
#             page_number=page_num + 1,
#             text=anchor_text,
#             input_tokens=0,
#             output_tokens=0,
#             is_fallback=True,
#             performance=performance_metrics,
#             metadata={
#                 "primary_language": None,
#                 "is_table": False,
#                 "is_diagram": False,
#                 "rotation": 0
#             }
#         )
#     except Exception as e:
#         logger.exception(f"Even fallback processing failed for {pdf_path}-{page_num}: {e}")
#         return PageResult(
#             page_number=page_num + 1,
#             text="ERROR: Failed to extract text from this page.",
#             is_fallback=True,
#             performance=performance_metrics
#         )

# # Process PDFs in parallel with controlled concurrency
# async def process_pdf(pdf_path: str, 
#                       first_page: int = 1, 
#                       last_page: int = None,
#                       max_concurrent_pages: int = 4,
#                       **kwargs) -> List[PageResult]:
#     """Process a PDF file using parallel page processing with SGLang."""
#     pdf_start = time.perf_counter()
    
#     try:
#         # Get total page count
#         page_count_start = time.perf_counter()
#         num_pages = total_pages(pdf_path)
#         page_count_end = time.perf_counter()
        
#         logger.info(f"Processing PDF {pdf_path} with {num_pages} pages")
        
#         # Validate and adjust page range
#         first_page = max(1, first_page)
#         if last_page is None or last_page > num_pages:
#             last_page = num_pages
        
#         # Convert to 0-based indexing for internal use
#         pages_to_process = range(first_page - 1, last_page)
#         total_pages_to_process = len(pages_to_process)
        
#         # Use semaphore to limit concurrent page processing
#         semaphore = asyncio.Semaphore(max_concurrent_pages)
        
#         async def process_with_semaphore(worker_id, page_num):
#             async with semaphore:
#                 return await process_page(
#                     worker_id=worker_id,
#                     pdf_path=pdf_path,
#                     page_num=page_num,
#                     **kwargs
#                 )
        
#         # Create tasks for all pages
#         tasks = []
#         for i, page_num in enumerate(pages_to_process):
#             worker_id = i % max_concurrent_pages  # Assign worker IDs
#             tasks.append(asyncio.create_task(
#                 process_with_semaphore(worker_id, page_num)
#             ))
        
#         # Wait for all tasks to complete
#         page_results = await asyncio.gather(*tasks, return_exceptions=True)
        
#         # Handle any exceptions in results
#         processed_results = []
#         for i, result in enumerate(page_results):
#             if isinstance(result, Exception):
#                 logger.error(f"Error processing page {pages_to_process[i]+1}: {result}")
#                 # Create error result
#                 processed_results.append(PageResult(
#                     page_number=pages_to_process[i] + 1,
#                     text=f"ERROR: Failed to process page: {str(result)}",
#                     is_fallback=True
#                 ))
#             else:
#                 processed_results.append(result)
        
#         # Sort by page number
#         processed_results.sort(key=lambda x: x.page_number)
        
#         # Calculate success rate
#         fallback_pages = sum(1 for r in processed_results if r.is_fallback)
#         success_rate = (total_pages_to_process - fallback_pages) / total_pages_to_process if total_pages_to_process > 0 else 0
        
#         pdf_end = time.perf_counter()
#         logger.info(f"PDF {pdf_path} processed in {pdf_end - pdf_start:.2f}s with {success_rate:.1%} success rate")
        
#         return processed_results
        
#     except Exception as e:
#         logger.exception(f"Failed to process PDF {pdf_path}: {e}")
#         raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")



#,cache_dir="/local/home/hfurquan/myProjects/Leaderboard/cache"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    global processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "allenai/olmOCR-7B-0225-preview", 
        torch_dtype=torch.bfloat16
    ).eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    # Create a semaphore to control worker access
    # We only allow one worker to move forward with requests, until the server has no more requests in its queue
    # This lets us get full utilization by having many workers, but also to be outputting dolma docs as soon as possible
    # As soon as one worker is no longer saturating the gpu, the next one can start sending requests
    # semaphore = asyncio.Semaphore(1)

    # await sglang_server_task(semaphore)

    # await sglang_server_ready()
    print("OLM OCR model loaded ✅")
    
    yield

    # Wait for server to stop
    # process_pool.shutdown(wait=False)

    # sglang_server.cancel()
    # logger.info("Work done")
    
    del model
    del processor
    torch.cuda.empty_cache()
    gc.collect()

olm_ocr = APIRouter(lifespan=lifespan)

# async def process_page(args, worker_id: int, pdf_orig_path: str, pdf_local_path: str, page_num: int) -> PageResult:
#     COMPLETION_URL = f"http://localhost:{SGLANG_SERVER_PORT}/v1/chat/completions"
#     MAX_RETRIES = 5

#     exponential_backoffs = 0
#     local_anchor_text_len = 6000
#     local_image_rotation = 0
#     attempt = 0
#     await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "started")

#     while attempt < MAX_RETRIES:
#         query = await build_page_query(pdf_local_path, page_num, 1024, local_anchor_text_len, image_rotation=local_image_rotation)

#         logger.info(f"Built page query for {pdf_orig_path}-{page_num}")

#         try:
#             status_code, response_body = await apost(COMPLETION_URL, json_data=query)

#             if status_code == 400:
#                 raise ValueError(f"Got BadRequestError from server: {response_body}, skipping this response")
#             elif status_code == 500:
#                 raise ValueError(f"Got InternalServerError from server: {response_body}, skipping this response")
#             elif status_code != 200:
#                 raise ValueError(f"Error http status {status_code}")

#             base_response_data = json.loads(response_body)

#             if base_response_data["usage"]["total_tokens"] > args.model_max_context:
#                 local_anchor_text_len = max(1, local_anchor_text_len // 2)
#                 logger.info(f"Reducing anchor text len to {local_anchor_text_len} for {pdf_orig_path}-{page_num}")
#                 raise ValueError("Response exceeded model_max_context, cannot use this response")

#             metrics.add_metrics(
#                 sglang_input_tokens=base_response_data["usage"].get("prompt_tokens", 0),
#                 sglang_output_tokens=base_response_data["usage"].get("completion_tokens", 0),
#             )

#             model_response_json = json.loads(base_response_data["choices"][0]["message"]["content"])
#             page_response = PageResponse(**model_response_json)

#             if not page_response.is_rotation_valid and attempt < MAX_RETRIES - 1:
#                 logger.info(
#                     f"Got invalid_page rotation for {pdf_orig_path}-{page_num} attempt {attempt}, retrying with {page_response.rotation_correction} rotation"
#                 )
#                 local_image_rotation = page_response.rotation_correction
#                 raise ValueError(f"invalid_page rotation for {pdf_orig_path}-{page_num}")

#             await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "finished")
#             return PageResult(
#                 pdf_orig_path,
#                 page_num,
#                 page_response,
#                 input_tokens=base_response_data["usage"].get("prompt_tokens", 0),
#                 output_tokens=base_response_data["usage"].get("completion_tokens", 0),
#                 is_fallback=False,
#             )
#         except (ConnectionError, OSError, asyncio.TimeoutError) as e:
#             logger.warning(f"Client error on attempt {attempt} for {pdf_orig_path}-{page_num}: {type(e)} {e}")

#             # Now we want to do exponential backoff, and not count this as an actual page retry
#             # Page retrys are supposed to be for fixing bad results from the model, but actual requests to sglang
#             # are supposed to work. Probably this means that the server is just restarting
#             sleep_delay = 10 * (2**exponential_backoffs)
#             exponential_backoffs += 1
#             logger.info(f"Sleeping for {sleep_delay} seconds on {pdf_orig_path}-{page_num} to allow server restart")
#             await asyncio.sleep(sleep_delay)
#         except asyncio.CancelledError:
#             logger.info(f"Process page {pdf_orig_path}-{page_num} cancelled")
#             await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "cancelled")
#             raise
#         except json.JSONDecodeError as e:
#             logger.warning(f"JSON decode error on attempt {attempt} for {pdf_orig_path}-{page_num}: {e}")
#             attempt += 1
#         except ValueError as e:
#             logger.warning(f"ValueError on attempt {attempt} for {pdf_orig_path}-{page_num}: {type(e)} - {e}")
#             attempt += 1
#         except Exception as e:
#             logger.exception(f"Unexpected error on attempt {attempt} for {pdf_orig_path}-{page_num}: {type(e)} - {e}")
#             attempt += 1

#     logger.error(f"Failed to process {pdf_orig_path}-{page_num} after {MAX_RETRIES} attempts.")
#     await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "errored")

#     return PageResult(
#         pdf_orig_path,
#         page_num,
#         PageResponse(
#             natural_text=get_anchor_text(pdf_local_path, page_num, pdf_engine="pdftotext"),
#             primary_language=None,
#             is_rotation_valid=True,
#             rotation_correction=0,
#             is_table=False,
#             is_diagram=False,
#         ),
#         input_tokens=0,
#         output_tokens=0,
#         is_fallback=True,
#     )


# async def process_pdf(args, worker_id: int, pdf_orig_path: str):
#     with tempfile.NamedTemporaryFile("wb+", suffix=".pdf") as tf:
#         try:
#             data = await asyncio.to_thread(lambda: get_s3_bytes_with_backoff(pdf_s3, pdf_orig_path))
#             tf.write(data)
#             tf.flush()
#         except ClientError as ex:
#             if ex.response["Error"]["Code"] == "NoSuchKey":
#                 logger.info(f"S3 File Not found, skipping it completely {pdf_orig_path}")
#                 return None
#             else:
#                 raise

#         try:
#             reader = PdfReader(tf.name)
#             num_pages = reader.get_num_pages()
#         except:
#             logger.exception(f"Could not count number of pages for {pdf_orig_path}, aborting document")
#             return None

#         logger.info(f"Got {num_pages} pages to do for {pdf_orig_path} in worker {worker_id}")

#         if args.apply_filter and get_pdf_filter().filter_out_pdf(tf.name):
#             logger.info(f"Filtering out pdf {pdf_orig_path}")
#             return None

#         # List to hold the tasks for processing each page
#         page_tasks = []
#         page_results = []

#         try:
#             async with asyncio.TaskGroup() as tg:
#                 for page_num in range(1, num_pages + 1):
#                     task = tg.create_task(process_page(args, worker_id, pdf_orig_path, tf.name, page_num))
#                     page_tasks.append(task)

#             # Collect the results from the entire task group, assuming no exceptions
#             page_results = [task.result() for task in page_tasks]

#             num_fallback_pages = sum(page_result.is_fallback for page_result in page_results)

#             if num_fallback_pages / num_pages > args.max_page_error_rate:
#                 logger.error(
#                     f"Document {pdf_orig_path} has {num_fallback_pages} fallback pages out of {num_pages} exceeding max_page_error_rate of {args.max_page_error_rate}, discarding document."
#                 )
#                 return None
#             elif num_fallback_pages > 0:
#                 logger.warning(
#                     f"Document {pdf_orig_path} processed with {num_fallback_pages} fallback pages out of {num_pages}, proceeding to build Dolma document."
#                 )

#             return build_dolma_document(pdf_orig_path, page_results)
#         except Exception as e:
#             # Check for ExceptionGroup with BrokenProcessPool
#             if isinstance(e, ExceptionGroup):
#                 broken_pool, other = e.split(BrokenProcessPool)
#                 if broken_pool is not None:  # Found at least one BrokenProcessPool
#                     logger.critical("Encountered BrokenProcessPool, exiting process.")
#                     sys.exit(1)

#             logger.exception(f"Exception in process_pdf for {pdf_orig_path}: {e}")
#             # You can't build a dolma doc with even 1 failed page, so just get out of here
#             # However, you don't want to propagate an exception higher up and cancel the entire work_group
#             return None


def process_page(page_num, pdf_path, temperature, dpi, max_new_tokens, num_return_sequences, device):
    """Process a single page and return the OCR text with performance metrics."""
    perf_metrics = {}
    total_start = time.perf_counter()
    
    # Render page to an image
    render_start = time.perf_counter()
    image_base64 = render_pdf_to_base64png(pdf_path, page_num, target_longest_image_dim=1024)
    render_end = time.perf_counter()
    perf_metrics["render_time"] = render_end - render_start
    
    # Get anchor text
    anchor_start = time.perf_counter()
    try:
        anchor_text = get_anchor_text(pdf_path, page_num, pdf_engine="pdfreport", target_length=4000)
        prompt = build_finetuning_prompt(anchor_text)
    except:
        prompt = ""
    anchor_end = time.perf_counter()
    perf_metrics["anchor_text_time"] = anchor_end - anchor_start
    
    # Build the full prompt
    prep_start = time.perf_counter()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ],
        }
    ]
    
    # Apply the chat template and processor
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    main_image = Image.open(BytesIO(base64.b64decode(image_base64)))
    inputs = processor(
        text=[text],
        images=[main_image],
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for (key, value) in inputs.items()}
    prep_end = time.perf_counter()
    perf_metrics["preprocessing_time"] = prep_end - prep_start
    
    # Generate the output
    inference_start = time.perf_counter()
    output = model.generate(
        **inputs,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        do_sample=True,
    )
    inference_end = time.perf_counter()
    perf_metrics["inference_time"] = inference_end - inference_start
    
    # Decode the output
    postproc_start = time.perf_counter()
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_length:]
    text_output = processor.tokenizer.batch_decode(
        new_tokens, skip_special_tokens=True
    )
    
    try:
        page_text = json.loads(text_output[0])['natural_text']
    except:
        page_text = text_output[0]
    postproc_end = time.perf_counter()
    perf_metrics["postprocessing_time"] = postproc_end - postproc_start
    
    total_end = time.perf_counter()
    perf_metrics["total_time"] = total_end - total_start
    
    return {
        "page_number": page_num + 1, 
        "text": page_text,
        "performance": perf_metrics
    }

@olm_ocr.get("/olmocr")
def olm(pdf_path: str, first_page: int = 1, last_page: int = None, temprature: float = 0.9, dpi:int = 256, max_new_tokens: int = 5000, num_return_sequences: int = 1):
    """
    Perform OCR on a specific page of the given PDF using Tesseract.
    
    Args:
        pdf_path: Path to the PDF file
        first_page: First Page number to OCR (1-based index)
        last_page: Last Page number to OCR (1-based index)
        temprature: The value used to control the randomness of the generated text
        max_new_tokens: The maximum number of tokens to generate
        num_return_sequences: The number of sequences to generate
    
    Returns:
        JSON response with OCR'd text for the specified page(s)
    """
    api_start = time.perf_counter()
    
    # Locate the PDF file
    pdf_locate_start = time.perf_counter()
    try:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pdf_path = os.path.join(script_dir, "pdfs", pdf_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error rendering PDF: " + str(e))
    pdf_locate_end = time.perf_counter()
    
    # Get the number of pages in the PDF
    page_count_start = time.perf_counter()
    num_pages = total_pages(pdf_path)
    page_count_end = time.perf_counter()

    print(f"Total number of pages of {pdf_path}: {num_pages}")
    
    # Configure processing parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = []

    # Process pages
    processing_start = time.perf_counter()
    for page in range(1, 4):  # num_pages
        page_start = time.perf_counter()
        output.append(process_page(
            page_num=page,
            pdf_path=pdf_path, 
            temperature=temprature, 
            dpi=dpi, 
            max_new_tokens=max_new_tokens, 
            num_return_sequences=num_return_sequences,
            device=device
        ))
        page_end = time.perf_counter()
        print(f"Page {page+1} processed in {page_end - page_start:.2f}s")
    
    processing_end = time.perf_counter()
    
    # Sort and prepare results
    output.sort(key=lambda x: x['page_number'])
    
    # Calculate overall performance metrics
    api_end = time.perf_counter()
    
    # Add overall performance metrics
    performance_summary = {
        "pdf_location_time": pdf_locate_end - pdf_locate_start,
        "page_counting_time": page_count_end - page_count_start,
        "processing_time": processing_end - processing_start,
        "total_api_time": api_end - api_start,
        "pages_processed": len(output),
        "avg_page_processing_time": (processing_end - processing_start) / max(1, len(output))
    }
    
    # Return with performance metrics
    return {
        "pages": output,
        "performance_summary": performance_summary
    }


    # Use ThreadPoolExecutor since we're primarily I/O bound with the model inference
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Create a partial function with the fixed parameters
        process_func = partial(
            process_page, 
            pdf_path=pdf_path, 
            temperature=temprature, 
            dpi=dpi, 
            max_new_tokens=max_new_tokens, 
            num_return_sequences=num_return_sequences,
            device=device
        )
        
        # Process all pages in parallel
        for result in executor.map(process_func, range(3)): #num_pages for entire pdf
            result_dict["pages"].append(result)
            full_text += f"\n\n--- PAGE {result['page']} ---\n\n{result['text']}"

    #Sort the pages by page number
    result_dict["pages"] = sorted(result_dict["pages"], key=lambda x: x["page"])
    result_dict["full_text"] = full_text
    
    return result_dict