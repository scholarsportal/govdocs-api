import base64
import io
from multiprocessing import Pool
import subprocess
from typing import List
import os
from pdf2image import convert_from_path, pdfinfo_from_path
from pathlib import Path

from PIL import Image
import platform

# Check the operating system platform
if platform.system() == "Windows":
    # For Windows, use the full path if poppler is installed at the expected location
    pdfinfo_path = "pdfinfo"
    pdftoppm_path = "pdftoppm"
    
    # Add Poppler executables to the PATH if installed in standard location
    poppler_path = "C:\\poppler\\poppler-24.08.0\\Library\\bin"
    if os.path.exists(poppler_path):
        os.environ["PATH"] = poppler_path + os.pathsep + os.environ["PATH"]
    else:
        # Fallback to full paths if directory doesn't exist
        pdfinfo_path = "C:\\poppler\\poppler-24.08.0\\Library\\bin\\pdfinfo.exe"
        pdftoppm_path = "C:\\poppler\\poppler-24.08.0\\Library\\bin\\pdftoppm.exe"
else:
    # For Linux/macOS, assume poppler-utils is installed via package manager
    pdfinfo_path = "pdfinfo"
    pdftoppm_path = "pdftoppm"

def get_pdf_media_box_width_height(local_pdf_path: str, page_num: int) -> tuple[float, float]:
    """
    Get the MediaBox dimensions for a specific page in a PDF file using the pdfinfo command.

    :param pdf_file: Path to the PDF file
    :param page_num: The page number for which to extract MediaBox dimensions
    :return: A dictionary containing MediaBox dimensions or None if not found
    """
    
    #pdfinfo_path = "C:\\poppler\\poppler-24.08.0\\Library\\bin\\pdfinfo.exe"
    # Construct the pdfinfo command to extract info for the specific page
    command = [pdfinfo_path, "-f", str(page_num), "-l", str(page_num), "-box", "-enc", "UTF-8", local_pdf_path]
    # "pdfinfo"
    # Run the command using subprocess
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Check if there is any error in executing the command
    if result.returncode != 0:
        raise ValueError(f"Error running pdfinfo: {result.stderr}")

    # Parse the output to find MediaBox
    output = result.stdout

    for line in output.splitlines():
        if "MediaBox" in line:
            media_box_str: List[str] = line.split(":")[1].strip().split()
            media_box: List[float] = [float(x) for x in media_box_str]
            return abs(media_box[0] - media_box[2]), abs(media_box[3] - media_box[1])

    raise ValueError("MediaBox not found in the PDF info.")


def render_pdf_to_base64png(local_pdf_path: str, page_num: int, target_longest_image_dim: int = 2048):
    print(f"Current working directory: {os.getcwd()}")
    print(f"Attempting to access: {local_pdf_path}")
    print(f"File path exists: {os.path.exists(local_pdf_path)}")
    longest_dim = max(get_pdf_media_box_width_height(local_pdf_path, page_num))

    # Use full path to pdftoppm for Windows
    #pdftoppm_path = "C:\\poppler\\poppler-24.08.0\\Library\\bin\\pdftoppm.exe"  # Adjust path as needed

    # Convert PDF page to PNG using pdftoppm
    pdftoppm_result = subprocess.run(
        [
            pdftoppm_path, #"pdftoppm",
            "-png",
            "-f",
            str(page_num),
            "-l",
            str(page_num),
            "-r",
            str(target_longest_image_dim * 72 / longest_dim),  # 72 pixels per point is the conversion factor
            local_pdf_path,
        ],
        timeout=120,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert pdftoppm_result.returncode == 0, pdftoppm_result.stderr
    return base64.b64encode(pdftoppm_result.stdout).decode("utf-8")





MAX_WORKERS = 16


def extract_images(pages: List[int], filepath: str, dpi: int) -> List[tuple[Image.Image, int]]:
    """
    Extract images from a PDF file for a list of pages.

    :param pages: List of page numbers to extract images from
    :param filepath: Path to the PDF file
    :param dpi: Resolution for the extracted images
    :return: List of tuples containing the extracted image and the page number
    """
    images = convert_from_path(filepath, dpi=dpi, first_page=pages[0], last_page=pages[-1])
    return [(img, page_num) for img, page_num in zip(images, pages)]

def extract_images_from_pdf(filepath: str, dpi: int = 256, first_page: int = 1, last_page: int = None) -> List[tuple[Image.Image, int]]:
    """
    Reads a PDF file and extracts images from the specified pages.

    :param filepath: Path to the PDF file
    :param dpi: Resolution for the extracted images
    :param first_page: First page number to extract images  (1-based index)
    :param last_page: Last page number to extract images from (1-based index)
    :return: List of tuples containing the extracted image and the page number
    """
    # set total_pages to last_page if it is not None
    if last_page is not None:
        total_pages = last_page
    else:
        total_pages = pdfinfo_from_path(pdf_path=filepath, poppler_path=poppler_path)['Pages']
    # Split the pages into chunks for parallel processing
    page_chunks = [
        list(range(i, min(i + (total_pages // MAX_WORKERS) + 1, total_pages + 1)))
        for i in range(1, total_pages + 1, (total_pages // MAX_WORKERS) + 1)
    ]
    # Use a multiprocessing Pool to process the chunks
    with Pool(MAX_WORKERS) as pool:
        results = pool.starmap(extract_images, [(pages, filepath, dpi) for pages in page_chunks])
    # Flatten the list of results since each chunk is processed separately
    images = [result[0] for result in results ] #for item in result
    return images

def total_pages(pdf_path: str) -> int:
    """
    Get the total number of pages in a PDF file.

    :param pdf_path: Path to the PDF file
    :return: Total number of pages in the PDF
    """
    return pdfinfo_from_path(pdf_path=pdf_path, poppler_path=poppler_path)['Pages']