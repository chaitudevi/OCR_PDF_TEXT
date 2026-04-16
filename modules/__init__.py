"""Compatibility exports for the Streamlit app."""

from ocr_engine import OCRError, SUPPORTED_LANGS, check_tesseract, ocr_page_generator
from pdf_processor import (
    PDFProcessingError,
    extract_native_text,
    is_text_pdf,
    pdf_to_images,
    validate_pdf,
)
from summarizer import SummarizationError, summarise
from text_cleaner import (
    char_count,
    clean_ocr_text,
    merge_pages,
    sentence_count,
    word_count,
)

__all__ = [
    "OCRError",
    "PDFProcessingError",
    "SUPPORTED_LANGS",
    "SummarizationError",
    "char_count",
    "check_tesseract",
    "clean_ocr_text",
    "extract_native_text",
    "is_text_pdf",
    "merge_pages",
    "ocr_page_generator",
    "pdf_to_images",
    "sentence_count",
    "summarise",
    "validate_pdf",
    "word_count",
]
