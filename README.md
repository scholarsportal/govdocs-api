# 📚 GovDocs API

> The backend API for evaluating OCR performance on government documents.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103.1-009688.svg)](https://fastapi.tiangolo.com)
[![Poetry](https://img.shields.io/badge/Poetry-1.6.1-4B8BBE.svg)](https://python-poetry.org/)

## 🔍 Project Description

GovDocs API is an evaluation platform for various OCR (Optical Character Recognition) services specialized for government documents. This project provides a unified interface to compare the performance, accuracy, and output formats of different OCR technologies through a simple API.

## ✨ Features

The API supports the following OCR engines:

| OCR Service | Output Formats | Description |
|-------------|----------------|-------------|
| 🧠 [olmOCR](https://github.com/allenai/olmocr) | Plain Text | AI-powered OCR by Allen AI |
| 📝 [Marker OCR](https://github.com/VikParuchuri/marker) | HTML, Markdown, JSON | Advanced document layout-preserving OCR |
| 🔤 [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) | Plain text | Open-source OCR engine |
| 📄 [Docling OCR](https://huggingface.co/ds4sd/SmolDocling-256M-preview) | Markdown | Structure-preserving document OCR |

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or later
- Git
- GPU recommended for some OCR models

### 📦 Installation

#### Step 1: Clone the repository

```bash
git clone https://github.com/scholarsportal/govdocs-api.git
cd govdocs-api
```

#### Step 2: Install system dependencies

<details>
<summary>📋 Ubuntu/Debian</summary>

```bash
sudo apt-get update
sudo apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    ttf-mscorefonts-installer \
    msttcorefonts \
    fonts-crosextra-caladea \
    fonts-crosextra-carlito \
    gsfonts \
    lcdf-typetools
```

</details>

<details>
<summary>🪟 Windows</summary>

1. **Install Poppler**:
   - Download [Release-24.08.0-0.zip](https://github.com/oschwartz10612/poppler-windows/releases/tag/v24.08.0-0)
   - Extract to `C:\poppler\`
   - Add `C:\poppler\bin` to your system PATH

2. **Install Tesseract OCR**:
   - Follow instructions at [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
   - Add Tesseract to your system PATH

</details>

<details>
<summary>🍎 macOS</summary>

```bash
brew install poppler tesseract
```

</details>

#### Step 3: Install Poetry

Poetry is used for dependency management and packaging in Python.

<details>
<summary>📋 Linux, macOS, Windows (WSL)</summary>

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

</details>

<details>
<summary>🪟 Windows (PowerShell)</summary>

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

If you installed Python through the Microsoft Store, use `python` instead of `py`.
</details>

Make sure Poetry is in your PATH:

- Unix: `$HOME/.local/bin`
- Windows: `%APPDATA%\Python\Scripts`

#### Step 4: Install project dependencies

```bash
poetry install
```

### ⚙️ Configuration

1. Set up Supabase:

Ensure Supabase is up and running as described in the Supabase project [here](https://gitlab.scholarsportal.info/ai-ml/supabase).

1. Create a `.env` file from the example:

```bash
cp .env.example .env
```

1. Update your `.env` file with the Supabase API URL and anon key displayed after running `supabase start` from the Supabase project.
1. Set HF_HOME to download models to a custom path

### 🚀 Running the API

#### Local Development

Includes change tracking.

```bash
poetry run fastapi dev src/govdocs_api/server.py
```

#### Production

```bash
poetry run fastapi run src/govdocs_api/server.py
```

#### Docker

```bash
docker compose up
```

The API will be available at [http://localhost:8000](http://localhost:8000)

## 📖 API Documentation

Once the server is running, access the interactive API documentation at:

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## 🔄 API Endpoints

The API provides the following endpoints:

- `/marker` - Process documents using Marker OCR
- `/olmocr` - Process documents using olmOCR
- `/smoldocling` - Process documents using Docling OCR
- `/tesseract` - Process documents using Tesseract OCR

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
