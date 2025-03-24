# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3-slim

EXPOSE 8000

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app

# Accept Microsoft font license non-interactively
RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections

# Add system dependencies for OlmOCR
# RUN apt-get update && apt-get install -y \
#     poppler-utils \
#     ttf-mscorefonts-installer \
#     fonts-crosextra-caladea \
#     fonts-crosextra-carlito \
#     gsfonts \
#     lcdf-typetools \
#     git \
#     build-essential

# Add system dependencies for pdf utilities
RUN apt-get update && apt-get install -y \
poppler-utils \
ttf-mscorefonts-installer \
fonts-crosextra-caladea \
fonts-crosextra-carlito \
gsfonts \
lcdf-typetools \
git \
build-essential \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Configure Poetry to not create a virtual environment inside the container
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-interaction --no-ansi

# Install OlmOCR
# RUN git clone https://github.com/allenai/olmocr.git /tmp/olmocr && \
#     cd /tmp/olmocr && \
#     pip install -e .
# It's added to poetry dependencies instead

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "-k", "uvicorn.workers.UvicornWorker", "src.govdocs_api.server:app"]
