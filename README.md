# 📄 PDF OCR & Summariser

A production-ready Streamlit application that extracts text from scanned (image-based) PDF documents using Tesseract OCR and generates concise AI-powered summaries using HuggingFace BART.

---

## ✨ Features

| Feature | Detail |
|---|---|
| **PDF ingestion** | Scanned *and* native-text PDFs |
| **Smart routing** | Direct text extraction for digital PDFs; OCR for scanned ones |
| **Multi-page support** | Processes every page, shows per-page progress |
| **Image pre-processing** | Greyscale → contrast boost → median denoising before OCR |
| **Multi-language OCR** | English, French, German, Spanish, Portuguese, Italian, Arabic, Chinese, Hindi |
| **Text cleaning** | Hyphen re-joining, noise removal, paragraph preservation |
| **Abstractive summary** | Facebook BART-large-CNN via HuggingFace Transformers |
| **Extractive fallback** | TF-based sentence scoring – no model download needed |
| **Download outputs** | Extracted text and summary as `.txt` |
| **Document statistics** | Word count, sentence count, paragraphs, noise reduction % |

---

## 🗂 Project Structure

```
pdf_ocr_app/
├── app.py                  # Streamlit UI + pipeline orchestration
├── requirements.txt        # Python dependencies
├── README.md
└── modules/
    ├── __init__.py         # Public re-exports
    ├── pdf_processor.py    # PDF validation + image rasterisation (PyMuPDF)
    ├── ocr_engine.py       # Tesseract OCR + image pre-processing
    ├── text_cleaner.py     # OCR noise removal + paragraph normalisation
    └── summarizer.py       # BART abstractive + TF extractive summarisation
```

---

## ⚙️ Prerequisites

### 1 · Tesseract OCR (required)

Tesseract must be installed at the OS level before the app will work.

**Ubuntu / Debian**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr

# Extra language packs (example: French + German)
sudo apt-get install -y tesseract-ocr-fra tesseract-ocr-deu
```

**macOS (Homebrew)**
```bash
brew install tesseract

# Extra languages
brew install tesseract-lang
```

**Windows**
1. Download the installer from the [UB-Mannheim releases page](https://github.com/UB-Mannheim/tesseract/wiki).
2. Run the installer and note the installation path (e.g. `C:\Program Files\Tesseract-OCR`).
3. Add that path to your `PATH` environment variable.
4. *Optional:* In `ocr_engine.py` set `pytesseract.pytesseract.tesseract_cmd` to the full `.exe` path.

Verify installation:
```bash
tesseract --version
```

### 2 · Python 3.10+

```bash
python --version   # should print 3.10 or higher
```

---

## 🚀 Quick Start

```bash
# 1. Clone / download the project
git clone https://github.com/your-org/pdf-ocr-summariser.git
cd pdf-ocr-summariser

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

---

## 🧩 Summarisation Backends

### HuggingFace Transformers (default / recommended)
Uses `facebook/bart-large-cnn`, an abstractive model fine-tuned on CNN/DailyMail.  
**First run:** downloads ~1.7 GB of model weights (cached for subsequent runs).  
Select **"transformers"** or **"auto"** in the sidebar.

### Extractive (fast fallback)
No model download.  Scores sentences by term frequency and picks the top 8.  
Select **"extractive"** in the sidebar, or the app falls back automatically if Transformers is unavailable.

---

## 🌍 Adding More OCR Languages

1. Install the Tesseract language pack:
   ```bash
   sudo apt-get install tesseract-ocr-<lang-code>   # e.g. jpn for Japanese
   ```
2. Add an entry to `SUPPORTED_LANGS` in `modules/ocr_engine.py`:
   ```python
   SUPPORTED_LANGS = {
       ...
       "Japanese": "jpn",
   }
   ```

---

## 🐳 Docker (optional)

```dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t pdf-ocr-app .
docker run -p 8501:8501 pdf-ocr-app
```

---

## 🛠 Configuration Reference

| Sidebar Option | Default | Effect |
|---|---|---|
| OCR Language | English | Tesseract language pack to use |
| Render DPI | 200 | Image resolution for rasterisation (higher = slower + more accurate) |
| Summarisation Backend | auto | `auto` / `transformers` / `extractive` |
| Pre-process images | ✅ On | Greyscale + contrast + denoise before OCR |
| Always use OCR | ☐ Off | Force OCR even on native-text PDFs |

---

## 📈 Performance Tips

- **Large PDFs (> 20 pages):** keep DPI at 200 and use the extractive summariser to avoid model memory pressure.
- **Degraded scans:** increase DPI to 300 and ensure the correct language pack is installed.
- **GPU acceleration:** if you have a CUDA-capable GPU, `pip install torch --index-url https://download.pytorch.org/whl/cu121` will dramatically speed up the BART model.

---

## 🤝 Contributing

Pull requests are welcome.  Key extension points:

- `modules/summarizer.py` – add an OpenAI GPT backend by implementing `summarise_openai()` and wiring it into the `summarise()` dispatcher.
- `modules/ocr_engine.py` – swap Tesseract for EasyOCR or PaddleOCR by replacing `ocr_image()`.
- `app.py` – add a "Compare backends" mode that runs both and displays results side-by-side.

---

## 📄 License

MIT – see `LICENSE` for details.
# OCR_PDF_TEXT
