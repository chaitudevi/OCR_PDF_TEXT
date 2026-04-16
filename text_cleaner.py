"""
text_cleaner.py
---------------
Normalises raw OCR output into clean, readable prose.

OCR engines produce artefacts such as:
  - Excessive newlines / form-feeds
  - Hyphenated line-breaks ("infor-\nmation")
  - Stray Unicode noise
  - Double spaces and mixed whitespace

This module removes those artefacts while preserving paragraph structure.
"""

import re
import unicodedata
from dataclasses import dataclass


@dataclass
class CleaningStats:
    """Metrics collected during the cleaning pass."""
    original_chars: int = 0
    cleaned_chars: int = 0
    word_count: int = 0
    paragraph_count: int = 0

    @property
    def reduction_pct(self) -> float:
        if self.original_chars == 0:
            return 0.0
        return round((1 - self.cleaned_chars / self.original_chars) * 100, 1)


# ── Compiled regex patterns (pre-compile for performance) ─────────────────────

# Hyphenated line-breaks: "infor-\nmation" → "information"
_HYPHEN_BREAK = re.compile(r"(\w+)-\n(\w+)")

# Soft hyphens embedded by some PDF generators
_SOFT_HYPHEN = re.compile(r"\u00ad")

# Single newline within a paragraph (not a paragraph boundary)
# Two or more newlines → paragraph boundary (kept as \n\n)
_SINGLE_NEWLINE = re.compile(r"(?<!\n)\n(?!\n)")

# Three or more consecutive newlines collapsed to two
_MULTI_NEWLINE = re.compile(r"\n{3,}")

# Control characters except tab and newline
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Non-printable Unicode categories (Cc = control, Cs = surrogates)
# We keep Cf (format) chars like soft-hyphen (already stripped above)
_UNICODE_NOISE = re.compile(r"[^\S\n\t ]+")   # collapse exotic whitespace to space

# Multiple spaces to a single space (within a line)
_MULTI_SPACE = re.compile(r"[ \t]{2,}")

# Lines that are just punctuation / single characters (OCR noise)
_JUNK_LINE = re.compile(r"^\s*[^\w\s]{1,3}\s*$", re.MULTILINE)

# Page headers/footers that OCR often picks up: "Page 1 of 5" patterns
_PAGE_MARKER = re.compile(r"\bPage\s+\d+\s+of\s+\d+\b", re.IGNORECASE)


def _normalise_unicode(text: str) -> str:
    """Decompose and then recompose Unicode to NFC normal form."""
    return unicodedata.normalize("NFC", text)


def clean_ocr_text(
    raw_text: str,
    *,
    remove_page_markers: bool = True,
    remove_junk_lines: bool = True,
    preserve_paragraphs: bool = True,
) -> tuple[str, CleaningStats]:
    """
    Clean raw OCR text and return the cleaned string plus diagnostic stats.

    Processing order matters – each step assumes the previous one ran.

    Parameters
    ----------
    raw_text : str
        The concatenated OCR output from all PDF pages.
    remove_page_markers : bool
        Strip "Page N of M" strings if found.
    remove_junk_lines : bool
        Remove lines containing only punctuation (OCR artefacts).
    preserve_paragraphs : bool
        Keep double-newlines as paragraph breaks; collapse single newlines.

    Returns
    -------
    tuple[str, CleaningStats]
        (cleaned text, stats object)
    """
    stats = CleaningStats(original_chars=len(raw_text))

    text = raw_text

    # 1. Unicode normalisation
    text = _normalise_unicode(text)

    # 2. Remove soft hyphens
    text = _SOFT_HYPHEN.sub("", text)

    # 3. Re-join hyphenated line-breaks BEFORE collapsing newlines
    text = _HYPHEN_BREAK.sub(r"\1\2", text)

    # 4. Remove control characters
    text = _CONTROL_CHARS.sub("", text)

    # 5. Collapse exotic Unicode whitespace to a regular space
    text = _UNICODE_NOISE.sub(" ", text)

    # 6. Collapse multiple spaces / tabs on a single line
    text = _MULTI_SPACE.sub(" ", text)

    # 7. Strip "Page N of M" artefacts
    if remove_page_markers:
        text = _PAGE_MARKER.sub("", text)

    # 8. Remove junk lines (single punctuation etc.)
    if remove_junk_lines:
        text = _JUNK_LINE.sub("", text)

    # 9. Handle newlines
    if preserve_paragraphs:
        # Single newlines within a paragraph → space
        text = _SINGLE_NEWLINE.sub(" ", text)
        # Collapse 3+ newlines to 2 (paragraph break)
        text = _MULTI_NEWLINE.sub("\n\n", text)
    else:
        text = text.replace("\n", " ")

    # 10. Final trim of each paragraph
    paragraphs = [p.strip() for p in text.split("\n\n")]
    paragraphs = [p for p in paragraphs if p]          # drop empty paragraphs
    text = "\n\n".join(paragraphs)

    # Collect stats
    stats.cleaned_chars = len(text)
    stats.word_count = len(text.split())
    stats.paragraph_count = len(paragraphs)

    return text, stats


def merge_pages(page_texts: list[str], separator: str = "\n\n---\n\n") -> str:
    """
    Combine per-page OCR results into a single document string.

    Parameters
    ----------
    page_texts : list[str]
        List of per-page text strings (in page order).
    separator : str
        String inserted between pages to visually delimit them.

    Returns
    -------
    str
        Merged document text.
    """
    non_empty = [t.strip() for t in page_texts if t.strip()]
    return separator.join(non_empty)


def word_count(text: str) -> int:
    """Return the number of whitespace-separated tokens in *text*."""
    return len(text.split())


def char_count(text: str) -> int:
    """Return character count excluding whitespace."""
    return len(text.replace(" ", "").replace("\n", ""))


def sentence_count(text: str) -> int:
    """Rough sentence count based on terminal punctuation."""
    return len(re.findall(r"[.!?]+", text))
