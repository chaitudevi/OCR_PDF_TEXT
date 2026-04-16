"""
pdf_processor.py
----------------
Handles PDF ingestion and conversion to images.
Supports both native-text PDFs and scanned/image-based PDFs.
Uses PyMuPDF (fitz) as the primary engine with pdf2image as fallback.
"""

import io
import logging
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)

# ── Resolution for rasterisation ─────────────────────────────────────────────
DEFAULT_DPI = 200          # Good balance between quality and speed
HIGH_DPI    = 300          # Use for small fonts or degraded scans
MAX_PAGES   = 100          # Safety cap to prevent OOM on huge documents


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be opened or rasterised."""


def validate_pdf(file_bytes: bytes) -> int:
    """
    Open the PDF and return its page count.

    Parameters
    ----------
    file_bytes : bytes
        Raw bytes of the uploaded PDF file.

    Returns
    -------
    int
        Number of pages in the document.

    Raises
    ------
    PDFProcessingError
        If the file is not a valid PDF or is password-protected.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise PDFProcessingError(f"Cannot open PDF: {exc}") from exc

    if doc.is_encrypted:
        raise PDFProcessingError(
            "This PDF is password-protected. Please provide an unlocked file."
        )

    page_count = len(doc)
    doc.close()

    if page_count == 0:
        raise PDFProcessingError("The PDF contains no pages.")

    return page_count


def is_text_pdf(file_bytes: bytes, sample_pages: int = 3) -> bool:
    """
    Heuristic: check whether the PDF already contains selectable text.
    If so, we can attempt direct extraction before falling back to OCR.

    Parameters
    ----------
    file_bytes : bytes
        Raw PDF bytes.
    sample_pages : int
        Number of pages to sample (checks up to this many from the start).

    Returns
    -------
    bool
        True if meaningful text was found on at least one sampled page.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for i, page in enumerate(doc):
            if i >= sample_pages:
                break
            text = page.get_text("text").strip()
            if len(text) > 50:          # > 50 chars means real text is likely present
                doc.close()
                return True
        doc.close()
    except Exception:
        pass
    return False


def pdf_to_images(
    file_bytes: bytes,
    dpi: int = DEFAULT_DPI,
    max_pages: int = MAX_PAGES,
) -> Generator[tuple[int, Image.Image], None, None]:
    """
    Convert each PDF page into a PIL Image, yielding (page_number, image) pairs.

    Parameters
    ----------
    file_bytes : bytes
        Raw PDF bytes.
    dpi : int
        Rendering resolution.  Higher → better OCR, slower processing.
    max_pages : int
        Hard cap on the number of pages to process.

    Yields
    ------
    tuple[int, PIL.Image.Image]
        (1-based page number, rendered PIL image)

    Raises
    ------
    PDFProcessingError
        On any rendering failure.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise PDFProcessingError(f"Failed to open PDF for rendering: {exc}") from exc

    total = min(len(doc), max_pages)
    logger.info("Rendering %d page(s) at %d DPI", total, dpi)

    zoom = dpi / 72.0          # PyMuPDF default is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(total):
        try:
            page = doc[page_num]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            yield page_num + 1, img           # 1-based page numbers for UX
        except Exception as exc:
            logger.warning("Could not render page %d: %s", page_num + 1, exc)
            # Yield a blank image so OCR can still produce an empty string
            yield page_num + 1, Image.new("RGB", (800, 1000), color="white")

    doc.close()


def extract_native_text(file_bytes: bytes) -> str:
    """
    Extract embedded (selectable) text from a text-based PDF without OCR.
    Useful as a fast path when the PDF is not scanned.

    Parameters
    ----------
    file_bytes : bytes
        Raw PDF bytes.

    Returns
    -------
    str
        Concatenated text from all pages.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text("text"))
        doc.close()
        return "\n\n".join(pages_text)
    except Exception as exc:
        logger.error("Native text extraction failed: %s", exc)
        return ""
