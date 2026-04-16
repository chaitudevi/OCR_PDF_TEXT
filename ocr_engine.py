"""
ocr_engine.py
-------------
Applies Tesseract OCR to PIL Images produced by pdf_processor.
Includes pre-processing steps that improve accuracy on scanned documents.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)


def _configure_tesseract_cmd() -> None:
    """Point pytesseract at common Windows install locations when PATH is stale."""
    if pytesseract.pytesseract.tesseract_cmd != "tesseract":
        return

    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
        Path(os.environ.get("PROGRAMFILES", "")) / "Tesseract-OCR" / "tesseract.exe",
        Path(os.environ.get("ProgramFiles(x86)", "")) / "Tesseract-OCR" / "tesseract.exe",
    ]

    for candidate in candidates:
        if candidate.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            return


_configure_tesseract_cmd()

# ── Tesseract configuration ───────────────────────────────────────────────────
# PSM 3  = Fully automatic page segmentation (default, works for most docs)
# PSM 6  = Assume a single uniform block of text (better for dense paragraphs)
# OEM 3  = Default engine mode (LSTM + legacy)
TESS_CONFIG_DEFAULT = "--psm 3 --oem 3"
TESS_CONFIG_SINGLE_BLOCK = "--psm 6 --oem 3"
TESS_CONFIG_SPARSE_TEXT = "--psm 11 --oem 3"

# Supported Tesseract language codes (extend as needed)
SUPPORTED_LANGS = {
    "English": "eng",
    "French": "fra",
    "German": "deu",
    "Spanish": "spa",
    "Portuguese": "por",
    "Italian": "ita",
    "Arabic": "ara",
    "Chinese Simplified": "chi_sim",
    "Hindi": "hin",
}


class OCRError(Exception):
    """Raised when Tesseract is unavailable or crashes."""


@dataclass(frozen=True)
class OCRResult:
    """OCR text plus the confidence score used to choose it."""

    text: str
    confidence: float
    config: str
    preprocessed: bool
    rotation: int


def check_tesseract() -> str:
    """
    Verify Tesseract is installed and return its version string.

    Raises
    ------
    OCRError
        If Tesseract cannot be found.
    """
    try:
        version = pytesseract.get_tesseract_version()
        return str(version)
    except Exception as exc:
        raise OCRError(
            "Tesseract OCR is not installed or not on your PATH.\n"
            "Install it with:\n"
            "  • Ubuntu/Debian: sudo apt-get install tesseract-ocr\n"
            "  • macOS:         brew install tesseract\n"
            "  • Windows:       https://github.com/UB-Mannheim/tesseract/wiki"
        ) from exc


def _mean_confidence(data: dict) -> float:
    confidences: list[float] = []
    for raw_conf in data.get("conf", []):
        try:
            conf = float(raw_conf)
        except (TypeError, ValueError):
            continue
        if conf >= 0:
            confidences.append(conf)
    return round(sum(confidences) / len(confidences), 1) if confidences else -1.0


def _recognised_word_count(data: dict) -> int:
    return sum(1 for text in data.get("text", []) if str(text).strip())


def _rotate_for_ocr(img: Image.Image, degrees: int) -> Image.Image:
    if degrees == 0:
        return img
    fill = 255 if img.mode == "L" else "white"
    return img.rotate(degrees, expand=True, fillcolor=fill)


def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Apply image-processing techniques that measurably improve OCR accuracy
    on typical scanned documents.

    Pipeline
    --------
    1. Convert to greyscale             – removes colour noise
    2. Slight sharpening                – improves character edges
    3. Contrast enhancement             – helps binarisation
    4. Light median filter              – reduces salt-and-pepper noise

    Parameters
    ----------
    img : PIL.Image.Image
        Original colour image of a PDF page.

    Returns
    -------
    PIL.Image.Image
        Pre-processed image ready for OCR.
    """
    # 1. Greyscale
    grey = img.convert("L")

    # 2. Sharpen edges
    sharpened = grey.filter(ImageFilter.SHARPEN)

    # 3. Boost contrast
    enhancer = ImageEnhance.Contrast(sharpened)
    high_contrast = enhancer.enhance(2.0)

    # 4. Median filter to smooth speckles without blurring text edges
    denoised = high_contrast.filter(ImageFilter.MedianFilter(size=3))

    return denoised


def ocr_image(
    img: Image.Image,
    lang: str = "eng",
    config: str = TESS_CONFIG_DEFAULT,
    preprocess: bool = True,
) -> str:
    """
    Run Tesseract OCR on a single PIL Image and return the extracted text.

    Parameters
    ----------
    img : PIL.Image.Image
        The image to process (one PDF page).
    lang : str
        Tesseract language code, e.g. ``"eng"``, ``"fra"``.
    config : str
        Tesseract configuration string.
    preprocess : bool
        Whether to apply the pre-processing pipeline first.

    Returns
    -------
    str
        Recognised text (may be empty if the image is blank or unreadable).
    """
    return ocr_image_with_confidence(
        img, lang=lang, config=config, preprocess=preprocess
    ).text


def ocr_image_with_confidence(
    img: Image.Image,
    lang: str = "eng",
    config: str = TESS_CONFIG_DEFAULT,
    preprocess: bool = True,
) -> OCRResult:
    """
    Run Tesseract OCR and return the best result from a few page layouts.

    Certificates and forms often OCR better as a single text block, while dense
    pages can prefer automatic segmentation. Scoring candidates by confidence
    avoids feeding a visibly worse OCR pass into the summariser.
    """
    image_variants: list[tuple[bool, Image.Image]] = []
    if preprocess:
        image_variants.append((True, preprocess_image(img)))
    image_variants.append((False, img))

    configs = [config]
    for candidate in (TESS_CONFIG_SINGLE_BLOCK, TESS_CONFIG_DEFAULT, TESS_CONFIG_SPARSE_TEXT):
        if candidate not in configs:
            configs.append(candidate)

    best: OCRResult | None = None
    best_score = float("-inf")
    last_error: Exception | None = None

    for was_preprocessed, candidate_img in image_variants:
        for rotation in (0, 90, 180, 270):
            rotated_img = _rotate_for_ocr(candidate_img, rotation)
            for candidate_config in configs:
                try:
                    data = pytesseract.image_to_data(
                        rotated_img,
                        lang=lang,
                        config=candidate_config,
                        output_type=pytesseract.Output.DICT,
                    )
                    confidence = _mean_confidence(data)
                    recognised_words = _recognised_word_count(data)
                    score = confidence + min(recognised_words, 250) * 0.02
                    if confidence < 0 or recognised_words == 0:
                        continue
                    if score > best_score:
                        best = OCRResult(
                            text="",
                            confidence=confidence,
                            config=candidate_config,
                            preprocessed=was_preprocessed,
                            rotation=rotation,
                        )
                        best_score = score
                except pytesseract.TesseractError as exc:
                    last_error = exc
                    logger.debug("Tesseract candidate failed: %s", exc)
                except Exception as exc:
                    last_error = exc
                    logger.debug("Unexpected OCR candidate error: %s", exc)

    if best is not None:
        source_img = preprocess_image(img) if best.preprocessed else img
        source_img = _rotate_for_ocr(source_img, best.rotation)
        try:
            text = pytesseract.image_to_string(source_img, lang=lang, config=best.config)
        except pytesseract.TesseractError as exc:
            logger.error("Tesseract error: %s", exc)
            raise OCRError(f"OCR failed: {exc}") from exc
        except Exception as exc:
            logger.error("Unexpected OCR error: %s", exc)
            raise OCRError(f"OCR encountered an unexpected error: {exc}") from exc

        logger.debug(
            "Selected OCR config %s (preprocessed=%s, rotation=%s, confidence=%s)",
            best.config,
            best.preprocessed,
            best.rotation,
            best.confidence,
        )
        return OCRResult(
            text=text,
            confidence=best.confidence,
            config=best.config,
            preprocessed=best.preprocessed,
            rotation=best.rotation,
        )

    try:
        text = pytesseract.image_to_string(img, lang=lang, config=config)
        return OCRResult(
            text=text,
            confidence=-1.0,
            config=config,
            preprocessed=preprocess,
            rotation=0,
        )
    except pytesseract.TesseractError as exc:
        logger.error("Tesseract error: %s", exc)
        raise OCRError(f"OCR failed: {exc}") from exc
    except Exception as exc:
        if last_error is not None:
            exc = last_error
        logger.error("Unexpected OCR error: %s", exc)
        raise OCRError(f"OCR encountered an unexpected error: {exc}") from exc


def ocr_page_generator(
    pages: Generator,
    lang: str = "eng",
    preprocess: bool = True,
    include_confidence: bool = False,
) -> Generator[tuple[int, str] | tuple[int, str, float], None, None]:
    """
    Yield OCR results page by page, allowing the caller to update progress
    incrementally (useful for Streamlit progress bars).

    Parameters
    ----------
    pages : Generator[tuple[int, PIL.Image.Image], None, None]
        Output of ``pdf_processor.pdf_to_images()``.
    lang : str
        Tesseract language code.
    preprocess : bool
        Whether to pre-process each image before OCR.

    Yields
    ------
    tuple[int, str]
        (1-based page number, extracted text for that page)
    """
    for page_num, img in pages:
        logger.debug("Running OCR on page %d", page_num)
        try:
            result = ocr_image_with_confidence(img, lang=lang, preprocess=preprocess)
            if include_confidence:
                yield page_num, result.text, result.confidence
            else:
                yield page_num, result.text
        except OCRError as exc:
            logger.warning("Page %d OCR failed: %s", page_num, exc)
            if include_confidence:
                yield page_num, "", -1.0
            else:
                yield page_num, ""          # Return empty string so pipeline continues


def get_page_confidence(img: Image.Image, lang: str = "eng") -> float:
    """
    Return the mean OCR confidence score (0–100) for a page image.
    Useful for flagging low-quality scans to the user.

    Parameters
    ----------
    img : PIL.Image.Image
        The page image.
    lang : str
        Tesseract language code.

    Returns
    -------
    float
        Average word-level confidence, or -1.0 on failure.
    """
    try:
        data = pytesseract.image_to_data(
            preprocess_image(img),
            lang=lang,
            output_type=pytesseract.Output.DICT,
        )
        return _mean_confidence(data)
    except Exception:
        return -1.0
