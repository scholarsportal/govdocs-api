FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

EXPOSE 8000

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Accept Microsoft font license non-interactively
RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections


# Add system dependencies for pdf utilities
RUN apt-get update && apt-get install -y \
poppler-utils \
fonts-liberation \
fonts-crosextra-caladea \
fonts-crosextra-carlito \
gsfonts \
lcdf-typetools \
git \
build-essential \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the application
COPY . /app

# Install Poetry
RUN pip install poetry

# Configure Poetry to not create a virtual environment inside the container
RUN poetry config virtualenvs.create false

# Install dependencies, excluding development dependencies
RUN poetry install --no-interaction --no-ansi

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
#CMD ["gunicorn", "--bind", "0.0.0.0:8000", "-k", "uvicorn.workers.UvicornWorker", "src.govdocs_api.server:app"]
#CMD ["uvicorn", "src.govdocs_api.server:app", "--host", "0.0.0.0", "--port", "8000"]
#CMD [ "poetry", "run", "fastapi", "run", "src/govdocs_api/server.py"]
CMD [ "poetry", "run", "python", "src/govdocs_api/server.py"]