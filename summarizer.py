"""
summarizer.py
-------------
Provides text summarisation via two backends:

1. HuggingFace Transformers (abstractive, best quality)
   Model: facebook/bart-large-cnn

2. TextRank Extractive (fast fallback – no model download needed)
   Graph-based sentence ranking with position weighting.
"""

import logging
import re
from heapq import nlargest
from typing import Literal

logger = logging.getLogger(__name__)

# ── HuggingFace model settings ────────────────────────────────────────────────
HF_MODEL      = "facebook/bart-large-cnn"
HF_MAX_WORDS  = 700       # ~900 tokens – safe margin under BART's 1024 limit
HF_MIN_OUTPUT = 60
HF_MAX_OUTPUT = 280

# ── Extractive settings ───────────────────────────────────────────────────────
EXTRACTIVE_SENTENCES = 8
TEXTRANK_DAMPING     = 0.85
TEXTRANK_ITERATIONS  = 50

BackendType = Literal["auto", "transformers", "extractive"]

_hf_pipeline = None   # cached so the model is only loaded once per session


class SummarizationError(Exception):
    """Raised when no summarisation method succeeds."""


# ════════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ════════════════════════════════════════════════════════════════════════════════

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might must can could of in on at to for "
    "with by from up about into through during before after above below "
    "between and but or nor not so yet both either neither just because "
    "as if when while although though since so that which who whom whose "
    "this that these those i me my myself we our ours ourselves you your "
    "yours he him his she her hers they them their theirs it its".split()
)


def _tokenise_sentences(text: str) -> list[str]:
    """
    Split text into sentences, handling common abbreviations and
    multi-line OCR output robustly.
    """
    # Collapse multiple whitespace / newlines to single space first
    text = re.sub(r"\s+", " ", text).strip()

    # Protect common abbreviations from being split
    abbreviations = r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|i\.e|e\.g|Fig|No|Vol|p|pp)\."
    text = re.sub(abbreviations, lambda m: m.group().replace(".", "<DOT>"), text, flags=re.IGNORECASE)

    # Split on sentence-ending punctuation followed by space + capital / digit
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"\'\(])", text)

    # Restore protected dots and filter short fragments
    sentences = []
    for part in parts:
        part = part.replace("<DOT>", ".").strip()
        if len(part.split()) >= 5:   # skip very short fragments
            sentences.append(part)

    return sentences


def _content_words(sentence: str) -> set[str]:
    return {
        w for w in re.findall(r"\b[a-zA-Z]{3,}\b", sentence.lower())
        if w not in _STOPWORDS
    }


# ════════════════════════════════════════════════════════════════════════════════
# TextRank extractive summariser
# ════════════════════════════════════════════════════════════════════════════════

def _jaccard(words_a: set[str], words_b: set[str]) -> float:
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _build_similarity_matrix(sentences: list[str]) -> list[list[float]]:
    n = len(sentences)
    word_sets = [_content_words(s) for s in sentences]
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = _jaccard(word_sets[i], word_sets[j])
    return matrix


def _pagerank(matrix: list[list[float]], damping: float = TEXTRANK_DAMPING, iterations: int = TEXTRANK_ITERATIONS) -> list[float]:
    n = len(matrix)
    if n == 0:
        return []
    scores = [1.0 / n] * n
    col_sums = [sum(matrix[r][c] for r in range(n)) for c in range(n)]
    for _ in range(iterations):
        new_scores = []
        for i in range(n):
            rank = (1 - damping) / n
            for j in range(n):
                if col_sums[i] > 0:
                    rank += damping * (matrix[j][i] / col_sums[i]) * scores[j]
            new_scores.append(rank)
        scores = new_scores
    return scores


def _position_weight(idx: int, total: int) -> float:
    """Boost sentences near the start and end of the document."""
    if total <= 1:
        return 1.0
    # Sentences in the first 15% or last 10% get a 40% bonus
    if idx < max(1, int(total * 0.15)) or idx >= total - max(1, int(total * 0.10)):
        return 1.4
    return 1.0


def summarise_extractive(text: str, n_sentences: int = EXTRACTIVE_SENTENCES) -> str:
    """
    TextRank-based extractive summarisation with position weighting.

    Parameters
    ----------
    text : str
        Cleaned document text.
    n_sentences : int
        Number of sentences to include in the summary.

    Returns
    -------
    str
        Selected sentences joined in their original document order.
    """
    sentences = _tokenise_sentences(text)
    if not sentences:
        raise SummarizationError("Could not split text into sentences.")

    n_sentences = min(n_sentences, len(sentences))

    if len(sentences) == 1:
        return sentences[0]

    # Build similarity matrix and run PageRank
    matrix = _build_similarity_matrix(sentences)
    scores = _pagerank(matrix)

    # Apply position weighting
    total = len(sentences)
    weighted = [s * _position_weight(i, total) for i, s in enumerate(scores)]

    # Pick top-N, restore original order
    top_indices = sorted(
        nlargest(n_sentences, range(total), key=lambda i: weighted[i])
    )
    return " ".join(sentences[i] for i in top_indices)


# ════════════════════════════════════════════════════════════════════════════════
# HuggingFace BART abstractive summariser
# ════════════════════════════════════════════════════════════════════════════════

def _load_hf_pipeline():
    global _hf_pipeline
    if _hf_pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            logger.info("Loading HuggingFace model: %s", HF_MODEL)
            _hf_pipeline = hf_pipeline(
                "summarization",
                model=HF_MODEL,
                tokenizer=HF_MODEL,
                framework="pt",
            )
            logger.info("Model loaded successfully.")
        except ImportError:
            raise SummarizationError(
                "transformers / torch not installed. Run: pip install transformers torch"
            )
        except Exception as exc:
            raise SummarizationError(f"Failed to load model '{HF_MODEL}': {exc}") from exc
    return _hf_pipeline


def _chunk_by_sentences(text: str, max_words: int = HF_MAX_WORDS) -> list[str]:
    """
    Split text into sentence-aligned chunks that fit within max_words.
    Never cuts a sentence in half.
    """
    sentences = _tokenise_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_count + word_count > max_words and current:
            chunks.append(" ".join(current))
            current = [sentence]
            current_count = word_count
        else:
            current.append(sentence)
            current_count += word_count

    if current:
        chunks.append(" ".join(current))
    return chunks


def summarise_transformers(text: str) -> str:
    """
    Abstractive summarisation using BART via HuggingFace Transformers.
    Long documents are split into sentence-aligned chunks; partial summaries
    are condensed in a second pass.
    """
    pipe = _load_hf_pipeline()
    chunks = _chunk_by_sentences(text)
    if not chunks:
        raise SummarizationError("Text is empty after chunking.")

    def _summarise_chunk(chunk: str, max_out: int = HF_MAX_OUTPUT, min_out: int = HF_MIN_OUTPUT) -> str:
        input_words = len(chunk.split())
        max_length = min(max_out, max(30, int(input_words * 0.60)))
        min_length = min(min_out, max(15, int(max_length * 0.30)))
        result = pipe(chunk, max_length=max_length, min_length=min_length, do_sample=False, truncation=True)
        return result[0]["summary_text"]

    partial: list[str] = []
    for chunk in chunks:
        try:
            partial.append(_summarise_chunk(chunk))
        except Exception as exc:
            logger.warning("Chunk summarisation failed: %s", exc)

    if not partial:
        raise SummarizationError("All chunk summarisations failed.")

    # Second pass: condense partial summaries into one coherent summary
    if len(partial) > 1:
        combined = " ".join(partial)
        try:
            return _summarise_chunk(combined, max_out=HF_MAX_OUTPUT, min_out=HF_MIN_OUTPUT)
        except Exception:
            pass   # fall through to joined partials

    return " ".join(partial)


# ════════════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════════════

def summarise(
    text: str,
    backend: BackendType = "auto",
    n_sentences: int = EXTRACTIVE_SENTENCES,
) -> tuple[str, str]:
    """
    Summarise text using the specified or best-available backend.

    Parameters
    ----------
    text : str
        Cleaned document text to summarise.
    backend : BackendType
        ``"auto"``          – try BART first, fall back to TextRank extractive.
        ``"transformers"``  – HuggingFace BART only (raises on failure).
        ``"extractive"``    – TextRank extractive summarisation.
    n_sentences : int
        For the extractive backend: number of sentences to extract.

    Returns
    -------
    tuple[str, str]
        (summary_text, backend_used)
    """
    if not text or not text.strip():
        raise SummarizationError("Cannot summarise empty text.")

    word_count = len(text.split())

    # Very short documents – just return extractive top-3
    if word_count < 120 and backend == "auto":
        summary = summarise_extractive(text, n_sentences=min(3, n_sentences))
        return summary, "extractive (short document)"

    if backend == "extractive":
        return summarise_extractive(text, n_sentences), "extractive"

    if backend == "transformers":
        return summarise_transformers(text), "transformers (BART)"

    # backend == "auto": try BART, fall back to TextRank
    try:
        return summarise_transformers(text), "transformers (BART)"
    except Exception as exc:
        logger.warning("Transformers backend unavailable (%s), using TextRank extractive.", exc)
        return summarise_extractive(text, n_sentences), "extractive (fallback)"
