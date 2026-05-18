FROM python:3.10-slim

WORKDIR /app

# System dependencies for sentence-transformers and matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project (models and input files mounted separately or pre-built)
COPY . .

# Pre-cache FinBERT model (downloaded to models/finbert/)
RUN python models/download_model.py

# Pre-build RAG embedding index
RUN python -c "from src.rag_retriever import RAGRetriever; RAGRetriever().ensure_index()"

# Generate synthetic datasets
RUN python generate_dataset.py && python src/health_dataset.py

# Default command: reproduce all results
CMD ["bash", "reproduce.sh"]
