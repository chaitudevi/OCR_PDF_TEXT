"""
app.py
------
Streamlit front-end for the PDF OCR & Summarisation application.

Run with:
    streamlit run app.py

Architecture
------------
  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │  PDF Upload  │ → │  pdf2images  │ → │  OCR Engine  │ → │  Summariser  │
  │  (Streamlit) │    │ (PyMuPDF)    │    │ (Tesseract)  │    │ (BART / TF)  │
  └─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
"""

import io
import logging
import sys
from pathlib import Path

import streamlit as st

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from modules import (
    PDFProcessingError,
    OCRError,
    SummarizationError,
    SUPPORTED_LANGS,
    char_count,
    check_tesseract,
    clean_ocr_text,
    extract_native_text,
    is_text_pdf,
    merge_pages,
    ocr_page_generator,
    pdf_to_images,
    sentence_count,
    summarise,
    validate_pdf,
    word_count,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF OCR & Summariser",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* ── Brand colours ── */
        :root {
            --primary:   #2563EB;   /* blue-600  */
            --primary-l: #EFF6FF;   /* blue-50   */
            --success:   #16A34A;   /* green-600 */
            --warning:   #D97706;   /* amber-600 */
            --danger:    #DC2626;   /* red-600   */
            --border:    #E2E8F0;
            --text-muted:#64748B;
        }

        /* ── Header banner ── */
        .app-header {
            background: linear-gradient(135deg, #1E40AF 0%, #2563EB 60%, #3B82F6 100%);
            padding: 2rem 2.5rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            color: white;
        }
        .app-header h1 { font-size: 2rem; margin: 0; font-weight: 800; }
        .app-header p  { font-size: 1rem; margin: .4rem 0 0; opacity: .85; }

        /* ── Cards ── */
        .stat-card {
            background: white;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,.06);
        }
        .stat-card .value { font-size: 1.6rem; font-weight: 700; color: var(--primary); }
        .stat-card .label { font-size: .8rem; color: var(--text-muted); margin-top: .2rem; }

        /* ── Status badges ── */
        .badge {
            display: inline-block;
            padding: .25rem .65rem;
            border-radius: 999px;
            font-size: .75rem;
            font-weight: 600;
            letter-spacing: .03em;
        }
        .badge-success { background:#DCFCE7; color:#15803D; }
        .badge-warning { background:#FEF3C7; color:#92400E; }
        .badge-info    { background:#DBEAFE; color:#1D4ED8; }

        /* ── Section headers ── */
        .section-header {
            font-size: 1rem;
            font-weight: 700;
            color: #1E293B;
            border-left: 4px solid var(--primary);
            padding-left: .65rem;
            margin: 1.5rem 0 .75rem;
        }

        /* ── Text areas ── */
        .stTextArea textarea {
            font-family: 'IBM Plex Mono', 'Menlo', monospace;
            font-size: .83rem;
            line-height: 1.55;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
        }

        /* ── Buttons ── */
        div[data-testid="stButton"] > button {
            border-radius: 10px;
            font-weight: 600;
            transition: all .2s;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: #F8FAFC;
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] div[data-testid="stWidgetLabel"],
        [data-testid="stSidebar"] div[data-testid="stRadio"] label,
        [data-testid="stSidebar"] div[data-testid="stCheckbox"] label {
            color: #0F172A !important;
        }
        [data-testid="stSidebar"] small,
        [data-testid="stSidebar"] div[data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
            color: #475569 !important;
        }
        [data-testid="stSidebar"] .badge-success { background:#DCFCE7; color:#15803D !important; }
        [data-testid="stSidebar"] .badge-warning { background:#FEF3C7; color:#92400E !important; }
        [data-testid="stSidebar"] .badge-info    { background:#DBEAFE; color:#1D4ED8 !important; }

        /* ── Spinner text ── */
        .stSpinner p { color: var(--primary) !important; font-weight: 500; }

        /* ── Divider ── */
        hr { border-color: var(--border); margin: 1.5rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session-state initialisation ──────────────────────────────────────────────
for key in ("extracted_text", "summary", "stats", "backend_used", "page_count", "warnings"):
    if key not in st.session_state:
        st.session_state[key] = None


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR – settings
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    # Tesseract status check
    try:
        tess_version = check_tesseract()
        st.markdown(
            f'<span class="badge badge-success">✓ Tesseract {tess_version}</span>',
            unsafe_allow_html=True,
        )
    except OCRError as e:
        st.error(str(e))

    st.markdown("---")

    # Language selector
    lang_label = st.selectbox(
        "📖 OCR Language",
        options=list(SUPPORTED_LANGS.keys()),
        index=0,
        help="Select the primary language of the scanned document.",
    )
    ocr_lang = SUPPORTED_LANGS[lang_label]

    # DPI selector
    dpi = st.select_slider(
        "🔍 Render DPI",
        options=[150, 200, 250, 300],
        value=300,
        help=(
            "Higher DPI = better OCR accuracy but slower processing. "
            "200 is a good default; use 300 for degraded or small-font scans."
        ),
    )

    # Summarisation backend
    st.markdown("---")
    sum_backend = st.radio(
        "🤖 Summarisation Backend",
        options=["auto", "transformers", "extractive"],
        index=0,
        help=(
            "**auto** – tries HuggingFace BART first, falls back to extractive.\n\n"
            "**transformers** – abstractive (requires ~1.7 GB download on first run).\n\n"
            "**extractive** – fast, no model download needed."
        ),
    )

    # Preprocessing toggle
    st.markdown("---")
    preprocess = st.toggle("🖼️ Pre-process images", value=True,
        help="Applies greyscale + contrast + denoising before OCR. "
             "Usually improves accuracy. Disable for already-clean scans.")

    force_ocr = st.toggle("🔄 Always use OCR", value=False,
        help="Force OCR even when the PDF contains embedded text. "
             "Useful if embedded text is garbled.")

    st.markdown("---")
    st.caption("PDF OCR & Summariser · v1.0.0")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
        <h1>📄 PDF OCR &amp; Summariser</h1>
        <p>Upload a scanned or digital PDF · Extract text with Tesseract · Generate an AI-powered summary</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── File upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Drop your PDF here, or click to browse",
    type=["pdf"],
    accept_multiple_files=False,
    help="Scanned (image-based) and native-text PDFs are both supported.",
)


def render_stats():
    """Render the document statistics row."""
    stats = st.session_state.stats
    if not stats:
        return

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        (c1, stats.get("pages", "—"),      "Pages"),
        (c2, f'{stats.get("words", 0):,}',  "Words"),
        (c3, stats.get("sentences", 0),     "Sentences"),
        (c4, stats.get("paragraphs", 0),    "Paragraphs"),
    ]
    for col, val, label in metrics:
        with col:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="value">{val}</div>'
                f'<div class="label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Process button ────────────────────────────────────────────────────────────
if uploaded_file is not None:
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        process_clicked = st.button(
            "🚀 Process Document",
            type="primary",
            use_container_width=True,
        )
    with col_info:
        st.caption(f"**{uploaded_file.name}** · {uploaded_file.size / 1024:.1f} KB")

    if process_clicked:
        file_bytes = uploaded_file.read()
        st.session_state.warnings = []

        # ── Step 1: Validate ──────────────────────────────────────────────────
        try:
            page_count = validate_pdf(file_bytes)
            st.session_state.page_count = page_count
        except PDFProcessingError as e:
            st.error(f"❌ {e}")
            st.stop()

        # ── Step 2: OCR or native extraction ─────────────────────────────────
        page_texts: list[str] = []
        page_confidences: list[float] = []
        progress_bar = st.progress(0, text="Starting…")
        status_box   = st.empty()

        try:
            use_ocr = force_ocr or not is_text_pdf(file_bytes)

            if not use_ocr:
                status_box.info("📋 Detected embedded text – using direct extraction (fast path).")
                with st.spinner("Extracting native text…"):
                    native = extract_native_text(file_bytes)
                    page_texts = native.split("\x0c")   # form-feed = page break in PyMuPDF
                progress_bar.progress(1.0, text="Extraction complete")

            else:
                status_box.info("🔍 Scanned PDF detected – running Tesseract OCR…")
                images_gen = pdf_to_images(file_bytes, dpi=dpi)

                with st.spinner(f"Running OCR on {page_count} page(s)…"):
                    for page_num, page_text, confidence in ocr_page_generator(
                        images_gen,
                        lang=ocr_lang,
                        preprocess=preprocess,
                        include_confidence=True,
                    ):
                        page_texts.append(page_text)
                        if confidence >= 0:
                            page_confidences.append(confidence)
                        pct = page_num / page_count
                        progress_bar.progress(
                            pct,
                            text=(
                                f"OCR · Page {page_num} of {page_count} "
                                f"· confidence {confidence:.1f}%"
                                if confidence >= 0
                                else f"OCR · Page {page_num} of {page_count}"
                            ),
                        )

                        if not page_text.strip():
                            st.session_state.warnings.append(
                                f"Page {page_num} produced no text (blank or unreadable)."
                            )
                        elif 0 <= confidence < 55:
                            st.session_state.warnings.append(
                                f"Page {page_num} OCR confidence is low ({confidence:.1f}%). "
                                "The summary may contain recognition errors."
                            )

        except OCRError as e:
            st.error(f"❌ OCR Error: {e}")
            st.stop()
        except PDFProcessingError as e:
            st.error(f"❌ PDF Error: {e}")
            st.stop()

        # ── Step 3: Merge + Clean ─────────────────────────────────────────────
        progress_bar.progress(0.85, text="Cleaning text…")
        raw_merged = merge_pages(page_texts)

        if not raw_merged.strip():
            st.error(
                "❌ No text could be extracted from this document. "
                "The PDF may contain only images with no recognisable characters, "
                "or the selected OCR language may be incorrect."
            )
            st.stop()

        cleaned_text, cleaning_stats = clean_ocr_text(raw_merged)
        avg_confidence = (
            round(sum(page_confidences) / len(page_confidences), 1)
            if page_confidences
            else None
        )
        if avg_confidence is not None and avg_confidence < 55:
            st.session_state.warnings.append(
                f"Average OCR confidence is low ({avg_confidence:.1f}%). "
                "Check the extracted text before relying on the summary."
            )
        st.session_state.extracted_text = cleaned_text
        st.session_state.stats = {
            "pages":      page_count,
            "words":      word_count(cleaned_text),
            "sentences":  sentence_count(cleaned_text),
            "paragraphs": cleaning_stats.paragraph_count,
            "reduction":  cleaning_stats.reduction_pct,
            "ocr_confidence": avg_confidence,
        }

        # ── Step 4: Summarise ─────────────────────────────────────────────────
        progress_bar.progress(0.92, text="Generating summary…")
        try:
            with st.spinner("Generating summary (this may take a moment for large documents)…"):
                summary, backend_used = summarise(cleaned_text, backend=sum_backend)
            st.session_state.summary = summary
            st.session_state.backend_used = backend_used
        except SummarizationError as e:
            st.warning(f"⚠️ Summarisation failed: {e}")
            st.session_state.summary = "(Summarisation unavailable)"
            st.session_state.backend_used = "none"

        progress_bar.progress(1.0, text="✅ Done!")
        status_box.success("Document processed successfully.")


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS SECTION
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.extracted_text:

    # ── Warnings ──────────────────────────────────────────────────────────────
    if st.session_state.warnings:
        with st.expander(f"⚠️ {len(st.session_state.warnings)} warning(s)"):
            for w in st.session_state.warnings:
                st.caption(f"• {w}")

    st.divider()

    # ── Stats row ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Document Statistics</div>', unsafe_allow_html=True)
    render_stats()

    st.divider()

    # ── Two-column results ────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-header">📝 Extracted Text</div>', unsafe_allow_html=True)

        stats = st.session_state.stats
        if stats:
            sub_cols = st.columns(3)
            sub_cols[0].caption(f"**{stats['words']:,}** words")
            if stats.get("ocr_confidence") is not None:
                sub_cols[1].caption(f"**{stats['ocr_confidence']}%** OCR confidence")
            else:
                sub_cols[1].caption(f"**{stats['reduction']}%** noise removed")
            sub_cols[2].caption(f"**{stats['paragraphs']}** paragraphs")

        st.text_area(
            label="extracted_text_area",
            value=st.session_state.extracted_text,
            height=480,
            label_visibility="collapsed",
        )

        # Download button
        st.download_button(
            label="⬇️ Download as .txt",
            data=st.session_state.extracted_text.encode("utf-8"),
            file_name="extracted_text.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col_right:
        st.markdown('<div class="section-header">💡 Summary</div>', unsafe_allow_html=True)

        if st.session_state.backend_used:
            badge_class = (
                "badge-success" if "transformers" in st.session_state.backend_used
                else "badge-warning" if "fallback" in st.session_state.backend_used
                else "badge-info"
            )
            st.markdown(
                f'<span class="badge {badge_class}">🤖 {st.session_state.backend_used}</span>',
                unsafe_allow_html=True,
            )

        summary_words = word_count(st.session_state.summary or "")
        st.caption(f"**{summary_words}** words")

        st.text_area(
            label="summary_text_area",
            value=st.session_state.summary or "",
            height=480,
            label_visibility="collapsed",
        )

        st.download_button(
            label="⬇️ Download summary as .txt",
            data=(st.session_state.summary or "").encode("utf-8"),
            file_name="summary.txt",
            mime="text/plain",
            use_container_width=True,
        )

elif uploaded_file is None:
    # ── Empty state ───────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; padding:3rem 1rem; color:#94A3B8;">
            <div style="font-size:4rem;">📂</div>
            <div style="font-size:1.1rem; font-weight:600; margin-top:1rem;">No document loaded</div>
            <div style="font-size:.9rem; margin-top:.5rem;">
                Upload a PDF using the uploader above to get started.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with [Streamlit](https://streamlit.io) · "
    "[PyMuPDF](https://pymupdf.readthedocs.io) · "
    "[Tesseract](https://tesseract-ocr.github.io) · "
    "[HuggingFace Transformers](https://huggingface.co/docs/transformers)"
)
