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

#### Install dependencies 

Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install poppler-utils ttf-mscorefonts-installer msttcorefonts fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools
```
Windows:

Download the poppler windows poppler-windows package instead: 

https://github.com/oschwartz10612/poppler-windows/releases/tag/v24.08.0-0 

1. Download the Release-24.08.0-0.zip file.
2. extract the file at C:\poppler\

#### Install poetry 

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

*Add Poetry to your PATH*

The installer creates a poetry wrapper in a well-known, platform-specific directory:

- $HOME/.local/bin on Unix.
- %APPDATA%\Python\Scripts on Windows.
- $POETRY_HOME/bin if $POETRY_HOME is set.

If this directory is not present in your $PATH, you can add it in order to invoke Poetry as poetry.

Alternatively, the full path to the poetry binary can always be used:

- ~/Library/Application Support/pypoetry/venv/bin/poetry on macOS.
- ~/.local/share/pypoetry/venv/bin/poetry on Linux/Unix.
- %APPDATA%\pypoetry\venv\Scripts\poetry on Windows.
- $POETRY_HOME/venv/bin/poetry if $POETRY_HOME is set.

## Install Tesseract-OCR package

Windows: 

follow instructions at https://github.com/UB-Mannheim/tesseract/wiki

Ubunutu/Linux: 

```
sudo apt install tesseract-ocr
```

## Setup enviornment variables & Initialize Supabase

1. Copy the .env.example file to a new .env file.

2. Install the supabase cli following these instructions: [Supabase CLI install](https://supabase.com/docs/guides/local-development/cli/getting-started)

3. From the project directory run:

```bash
supabase start
```

This will setup a local supabase instance on your machine. Copy the `API URL` and `anon key` and paste it in the .env file you created.

### Install required packages

Install the required packages:

```bash
poetry install
```

Run the server:

```bash
poetry run fastapi dev src/govdocs_api/server.py # run the api server in development mode
poetry run fastapi run src/govdocs_api/server.py # run the api server in production mode
```

### Run with Docker

```bash
docker compose up
```

The API will be available on <http://localhost:8000>
