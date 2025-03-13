import base64
import io
import subprocess
from typing import List
import os
from pdf2image import convert_from_path
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


def render_pdf_to_base64webp(local_pdf_path: str, page: int, target_longest_image_dim: int = 1024):
    base64_png = render_pdf_to_base64png(local_pdf_path, page, target_longest_image_dim)

    png_image = Image.open(io.BytesIO(base64.b64decode(base64_png)))
    webp_output = io.BytesIO()
    png_image.save(webp_output, format="WEBP")

    return base64.b64encode(webp_output.getvalue()).decode("utf-8")


def get_png_dimensions_from_base64(base64_data) -> tuple[int, int]:
    """
    Returns the (width, height) of a PNG image given its base64-encoded data,
    without base64-decoding the entire data or loading the PNG itself

    Should be really fast to support filtering

    Parameters:
    - base64_data (str): Base64-encoded PNG image data.

    Returns:
    - tuple: (width, height) of the image.

    Raises:
    - ValueError: If the data is not a valid PNG image or the required bytes are not found.
    """
    # PNG signature is 8 bytes
    png_signature_base64 = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")
    if not base64_data.startswith(png_signature_base64[:8]):
        raise ValueError("Not a valid PNG file")

    # Positions in the binary data where width and height are stored
    width_start = 16  # Byte position where width starts (0-based indexing)
    _width_end = 20  # Byte position where width ends (exclusive)
    _height_start = 20
    height_end = 24

    # Compute the byte range needed (from width_start to height_end)
    start_byte = width_start
    end_byte = height_end

    # Calculate base64 character positions
    # Each group of 3 bytes corresponds to 4 base64 characters
    base64_start = (start_byte // 3) * 4
    base64_end = ((end_byte + 2) // 3) * 4  # Add 2 to ensure we cover partial groups

    # Extract the necessary base64 substring
    base64_substring = base64_data[base64_start:base64_end]

    # Decode only the necessary bytes
    decoded_bytes = base64.b64decode(base64_substring)

    # Compute the offset within the decoded bytes
    offset = start_byte % 3

    # Extract width and height bytes
    width_bytes = decoded_bytes[offset : offset + 4]
    height_bytes = decoded_bytes[offset + 4 : offset + 8]

    if len(width_bytes) < 4 or len(height_bytes) < 4:
        raise ValueError("Insufficient data to extract dimensions")

    # Convert bytes to integers
    width = int.from_bytes(width_bytes, "big")
    height = int.from_bytes(height_bytes, "big")

    return width, height

# Helper function to convert PDF to images
def convert_pdf_to_images(
  pdf_path,
  dpi=200,
  output_folder=None,
  first_page=None,
  last_page=None,
  fmt="ppm",
  jpegopt=None,
  thread_count=1,
  userpw=None,
  ownerpw=None,
  use_cropbox=False,
  strict=False,
  transparent=False,
  single_file=False,
  output_file=None,
  grayscale=False,
  size=None,
  paths_only=False,
  use_pdftocairo=False,
  timeout=None,
  hide_annotations=False,
):
  """
    Convert PDF to a list of PIL images with configurable parameters.
    
    :param pdf_path: Path to the PDF that you want to convert
    :type pdf_path: Union[str, PurePath]
    :param dpi: Image quality in DPI (default 200), defaults to 200
    :type dpi: int, optional
    :param output_folder: Write the resulting images to a folder (instead of directly in memory), defaults to None
    :type output_folder: Union[str, PurePath], optional
    :param first_page: First page to process, defaults to None
    :type first_page: int, optional
    :param last_page: Last page to process before stopping, defaults to None
    :type last_page: int, optional
    :param fmt: Output image format, defaults to "ppm"
    :type fmt: str, optional
    :param jpegopt: jpeg options `quality`, `progressive`, and `optimize` (only for jpeg format), defaults to None
    :type jpegopt: Dict, optional
    :param thread_count: How many threads we are allowed to spawn for processing, defaults to 1
    :type thread_count: int, optional
    :param userpw: PDF's password, defaults to None
    :type userpw: str, optional
    :param ownerpw: PDF's owner password, defaults to None
    :type ownerpw: str, optional
    :param use_cropbox: Use cropbox instead of mediabox, defaults to False
    :type use_cropbox: bool, optional
    :param strict: When a Syntax Error is thrown, it will be raised as an Exception, defaults to False
    :type strict: bool, optional
    :param transparent: Output with a transparent background instead of a white one, defaults to False
    :type transparent: bool, optional
    :param single_file: Uses the -singlefile option from pdftoppm/pdftocairo, defaults to False
    :type single_file: bool, optional
    :param output_file: What is the output filename or generator, defaults to uuid_generator()
    :type output_file: Any, optional
    :param poppler_path: Path to look for poppler binaries, defaults to None
    :type poppler_path: Union[str, PurePath], optional
    :param grayscale: Output grayscale image(s), defaults to False
    :type grayscale: bool, optional
    :param size: Size of the resulting image(s), uses the Pillow (width, height) standard, defaults to None
    :type size: Union[Tuple, int], optional
    :param paths_only: Don't load image(s), return paths instead (requires output_folder), defaults to False
    :type paths_only: bool, optional
    :param use_pdftocairo: Use pdftocairo instead of pdftoppm, may help performance, defaults to False
    :type use_pdftocairo: bool, optional
    :param timeout: Raise PDFPopplerTimeoutError after the given time, defaults to None
    :type timeout: int, optional
    :param hide_annotations: Hide PDF annotations in the output, defaults to False
    :type hide_annotations: bool, optional
    :raises NotImplementedError: Raised when conflicting parameters are given (hide_annotations for pdftocairo)
    :raises PDFPopplerTimeoutError: Raised after the timeout for the image processing is exceeded
    :raises PDFSyntaxError: Raised if there is a syntax error in the PDF and strict=True
    :return: A list of Pillow images, one for each page between first_page and last_page
    :rtype: List[Image.Image]
  """
  kwargs = {
    'dpi': dpi,
    'output_folder': output_folder,
    'first_page': first_page,
    'last_page': last_page,
    'fmt': fmt,
    'jpegopt': jpegopt,
    'thread_count': thread_count,
    'userpw': userpw,
    'ownerpw': ownerpw,
    'use_cropbox': use_cropbox,
    'strict': strict,
    'transparent': transparent,
    'single_file': single_file,
    'grayscale': grayscale,
    'size': size,
    'paths_only': paths_only,
    'use_pdftocairo': use_pdftocairo,
    'timeout': timeout,
    'hide_annotations': hide_annotations,
  }
  
  # Only add output_file if it's not None
  if output_file is not None:
    kwargs['output_file'] = output_file
  
  if platform.system() == "Windows":
    kwargs['poppler_path'] = Path("C:\\poppler\\poppler-24.08.0\\Library\\bin\\")
  
  return convert_from_path(pdf_path, **kwargs)