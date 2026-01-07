# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY master.py .

# Set environment variables (will be overridden by Azure)
ENV AZURE_STORAGE_CONNECTION_STRING=""
ENV GOOGLE_API_KEY=""

# Run the pipeline
CMD ["python", "-u", "master.py"]
