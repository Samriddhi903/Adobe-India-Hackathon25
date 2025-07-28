FROM python:3.10-slim

# Create directory structure first
RUN mkdir -p \
    /app/models/huggingface \
    /app/models/sentence_transformers \
    /app/models/torch \
    /app/nltk_data

WORKDIR /app

# Install requirements first (without offline restrictions)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Temporarily allow online access during build
# Download models with network access (simplified single-line command)
RUN python -c "import os; os.environ['HF_HUB_OFFLINE'] = '0'; \
import nltk; \
nltk.download('punkt', download_dir='/app/nltk_data', quiet=True); \
nltk.download('punkt_tab', download_dir='/app/nltk_data', quiet=True); \
nltk.download('stopwords', download_dir='/app/nltk_data', quiet=True); \
nltk.download('wordnet', download_dir='/app/nltk_data', quiet=True); \
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
tokenizer = AutoTokenizer.from_pretrained('typeform/distilbert-base-uncased-mnli'); \
model = AutoModelForSequenceClassification.from_pretrained('typeform/distilbert-base-uncased-mnli'); \
tokenizer.save_pretrained('/app/models/huggingface'); \
model.save_pretrained('/app/models/huggingface'); \
from sentence_transformers import SentenceTransformer; \
embedding_model = SentenceTransformer('all-MiniLM-L6-v2'); \
embedding_model.save('/app/models/sentence_transformers/all-MiniLM-L6-v2')"

# Now set the environment variables for runtime offline usage
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV HF_EVALUATE_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV TORCH_HOME=/app/models/torch
ENV TRANSFORMERS_CACHE=/app/models/huggingface
ENV HF_HOME=/app/models/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/models/sentence_transformers
ENV NLTK_DATA=/app/nltk_data

# Copy application code
COPY . .

CMD ["python", "process_documents.py"]
