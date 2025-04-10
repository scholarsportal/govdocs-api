FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

EXPOSE 8000

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Add system dependencies for pdf utilities
RUN apt-get update && apt-get install -y \
poppler-utils \
git \
build-essential \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the application
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD [ "fastapi", "run", "src/govdocs_api/server.py"]
