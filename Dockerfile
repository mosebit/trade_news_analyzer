FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (needed for parser_edisclosure_playwright)
RUN playwright install --with-deps

# Copy application code
COPY . .

# Create directories for database storage
RUN mkdir -p /app/chroma_db_new

# Default command - can be overridden
CMD ["python", "llm_prediction/predictor_main.py"]