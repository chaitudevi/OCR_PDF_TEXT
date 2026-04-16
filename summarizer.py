"""
summarizer.py
-------------
Provides text summarisation via three backends, tried in priority order:

1. HuggingFace Transformers (abstractive, best quality)
   Model: facebook/bart-large-cnn  – widely used, good for English documents.

2. Simple Extractive (fallback – no external model needed)
   Scores sentences by term frequency and picks the top-N.

The caller sets ``backend="auto"`` (default) to let this module pick the best
available option, or passes ``"transformers"`` / ``"extractive"`` explicitly.
"""

import logging
import re
from heapq import nlargest
from typing import Literal

logger = logging.getLogger(__name__)

# ── HuggingFace model settings ────────────────────────────────────────────────
HF_MODEL      = "facebook/bart-large-cnn"
HF_MAX_INPUT  = 1024        # tokens – BART limit
HF_MIN_OUTPUT = 80          # words
HF_MAX_OUTPUT = 300         # words (≈ 200-280 tokens for English)

# ── Extractive settings ───────────────────────────────────────────────────────
EXTRACTIVE_SENTENCES = 8    # top sentences to extract

BackendType = Literal["auto", "transformers", "extractive"]

# Cached pipeline so we only load the model once per session
_hf_pipeline = None


class SummarizationError(Exception):
    """Raised when no summarisation method succeeds."""


# ── HuggingFace backend ───────────────────────────────────────────────────────

def _load_hf_pipeline():
    """Lazy-load the HuggingFace summarisation pipeline."""
    global _hf_pipeline
    if _hf_pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            logger.info("Loading HuggingFace model: %s", HF_MODEL)
            _hf_pipeline = hf_pipeline(
                "summarization",
                model=HF_MODEL,
                tokenizer=HF_MODEL,
                framework="pt",            # PyTorch
            )
            logger.info("Model loaded successfully.")
        except ImportError:
            raise SummarizationError(
                "transformers / torch not installed. "
                "Run: pip install transformers torch"
            )
        except Exception as exc:
            raise SummarizationError(
                f"Failed to load HuggingFace model '{HF_MODEL}': {exc}"
            ) from exc
    return _hf_pipeline


def _chunk_text(text: str, max_tokens: int = HF_MAX_INPUT) -> list[str]:
    """
    Split text into chunks that fit within the model's token window.
    Uses a word-count proxy (1 token ≈ 0.75 words for English).
    """
    max_words = int(max_tokens * 0.75)
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def summarise_transformers(text: str) -> str:
    """
    Abstractive summarisation using BART via HuggingFace Transformers.

    Splits long documents into chunks and summarises each chunk, then
    optionally condenses the partial summaries into a final summary.

    Parameters
    ----------
    text : str
        Cleaned document text.

    Returns
    -------
    str
        Abstractive summary.
    """
    pipe = _load_hf_pipeline()
    chunks = _chunk_text(text)
    if not chunks:
        raise SummarizationError("Text is empty after chunking.")

    partial_summaries = []
    for chunk in chunks:
        try:
            input_words = len(chunk.split())
            max_length = min(HF_MAX_OUTPUT, max(30, int(input_words * 0.65)))
            min_length = min(HF_MIN_OUTPUT, max(12, int(max_length * 0.35)))
            result = pipe(
                chunk,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True,
            )
            partial_summaries.append(result[0]["summary_text"])
        except Exception as exc:
            logger.warning("Chunk summarisation failed: %s", exc)

    if not partial_summaries:
        raise SummarizationError("All chunk summarisations failed.")

    # If there were multiple chunks, do a second-pass condensation
    if len(partial_summaries) > 1:
        combined = " ".join(partial_summaries)
        try:
            input_words = len(combined.split())
            max_length = min(HF_MAX_OUTPUT, max(40, int(input_words * 0.65)))
            min_length = min(HF_MIN_OUTPUT, max(15, int(max_length * 0.35)))
            final = pipe(
                combined,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True,
            )
            return final[0]["summary_text"]
        except Exception:
            pass   # Fall back to just joining partial summaries

    return " ".join(partial_summaries)


# ── Extractive fallback backend ───────────────────────────────────────────────

# Common English stop words (keep lightweight – no NLTK dependency)
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might must can could of in on at to for "
    "with by from up about into through during before after above below "
    "between and but or nor not so yet both either neither just because "
    "as if when while although though since so that which who whom whose "
    "this that these those i me my myself we our ours ourselves you your "
    "yours he him his she her hers they them their theirs it its".split()
)


def _sentence_tokenise(text: str) -> list[str]:
    """Split text into sentences using punctuation heuristics."""
    # Split on . ! ? followed by whitespace and a capital letter or end-of-string
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'(])", text)
    # Clean up any very short fragments
    return [s.strip() for s in raw if len(s.strip()) > 20]


def _term_frequency(sentences: list[str]) -> dict[str, float]:
    """Compute normalised term frequency across all sentences."""
    freq: dict[str, int] = {}
    for sentence in sentences:
        for word in re.findall(r"\b[a-zA-Z]{3,}\b", sentence.lower()):
            if word not in _STOPWORDS:
                freq[word] = freq.get(word, 0) + 1

    max_freq = max(freq.values()) if freq else 1
    return {w: c / max_freq for w, c in freq.items()}


def summarise_extractive(text: str, n_sentences: int = EXTRACTIVE_SENTENCES) -> str:
    """
    Extractive summarisation: pick the highest-scoring sentences by TF.

    Parameters
    ----------
    text : str
        Cleaned document text.
    n_sentences : int
        Number of sentences to include in the summary.

    Returns
    -------
    str
        Extracted sentences, in original document order.
    """
    sentences = _sentence_tokenise(text)
    if not sentences:
        raise SummarizationError("Could not tokenise text into sentences.")

    n_sentences = min(n_sentences, len(sentences))
    tf = _term_frequency(sentences)

    # Score each sentence
    scores: dict[int, float] = {}
    for idx, sentence in enumerate(sentences):
        words = re.findall(r"\b[a-zA-Z]{3,}\b", sentence.lower())
        score = sum(tf.get(w, 0) for w in words if w not in _STOPWORDS)
        scores[idx] = score / max(len(words), 1)    # normalise by sentence length

    # Pick top-N sentence indices, then restore original order
    top_indices = sorted(nlargest(n_sentences, scores, key=scores.get))
    summary_sentences = [sentences[i] for i in top_indices]

    return " ".join(summary_sentences)


# ── Public API ────────────────────────────────────────────────────────────────

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
        ``"auto"``          – try transformers first, fall back to extractive.
        ``"transformers"``  – HuggingFace BART only (raises on failure).
        ``"extractive"``    – Fast TF-based extractive summarisation.
    n_sentences : int
        For the extractive backend: number of sentences to extract.

    Returns
    -------
    tuple[str, str]
        (summary_text, backend_used)
        ``backend_used`` is the name of the backend that produced the result.

    Raises
    ------
    SummarizationError
        If the requested backend fails and no fallback is available.
    """
    if not text or not text.strip():
        raise SummarizationError("Cannot summarise empty text.")

    if len(text.split()) < 120 and backend == "auto":
        summary = summarise_extractive(text, n_sentences=min(n_sentences, 3))
        return summary, "extractive (short document)"

    if backend == "extractive":
        summary = summarise_extractive(text, n_sentences)
        return summary, "extractive"

    if backend == "transformers":
        summary = summarise_transformers(text)
        return summary, "transformers (BART)"

    # backend == "auto"
    try:
        summary = summarise_transformers(text)
        return summary, "transformers (BART)"
    except Exception as exc:
        logger.warning("Transformers backend unavailable (%s), using extractive.", exc)
        summary = summarise_extractive(text, n_sentences)
        return summary, "extractive (fallback)"
