# GovDocs API

The backend API for the OCUL Government Documents AIML project.

## Project description

The API will provide the ability to run and evaluate these OCR services:

* [GOT_OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0) (outputs HTML with a custom JS converter)
* [olmOCR](https://github.com/allenai/olmocr) ([Dolma](https://github.com/allenai/dolma)-style JSONL, uses dolmaviewer)
* [Marker OCR](https://github.com/VikParuchuri/marker) (Markdown)
* [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (Plain text)

## Run the project

This project requires Python 3.10 or later to run. You need to clone the project, install poetry, and then install dependencies before running the project:

```bash
git clone https://github.com/scholarsportal/govdocs-api.git
cd govdocs-api
```

### Run locally

```bash
pip install poetry
poetry install
poetry run uvicorn src.govdocs_api.server:app --reload
```

### Run with Docker

```bash
docker compose up
```

The API will be available on <http://localhost:8000>
