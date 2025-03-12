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

### Install poetry 

Poetry is a tool for dependency management and packaging in Python.

Follow installation instructions at: https://python-poetry.org/docs/#installing-with-the-official-installer

Linux, macOS, Windows (WSL)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
Windows (Powershell)
```bash
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```
If you have installed Python through the Microsoft Store, replace py with python in the command above.

### Install required packages
```bash
poetry config virtualenvs.in-project true # Allow poetry to create a virual env within project directory
poetry install
poetry run fastapi dev src/govdocs_api/server.py # run the api server in development mode
poetry run fastapi run src/govdocs_api/server.py # run the api server in production mode
```

### Run with Docker

```bash
docker compose up
```

The API will be available on <http://localhost:8000>
