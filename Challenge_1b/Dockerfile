FROM python:3.10-slim

# Set working directory
WORKDIR /app/Challenge_1b

# Copy requirements.txt for dependencies
COPY requirements.txt /app/requirements.txt

# Install dependencies (assumes offline build with cached packages or local wheelhouse)
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy Challenge_1b project files including models and collections
COPY . /app/Challenge_1b

# Set environment variables for model cache paths inside container
ENV TORCH_HOME=/app/models/torch
ENV SENTENCE_TRANSFORMERS_HOME=/app/models/sentence_transformers
ENV TRANSFORMERS_CACHE=/app/models/huggingface/hub

# Expose any ports if needed (not specified here)

# Default command to run the main processing script
CMD ["python", "process_documents.py"]
