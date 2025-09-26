# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY ../requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY ../src/ ./src/
COPY ../data/ ./data/
COPY ../airflow_dags/ ./airflow_dags/

# Create necessary directories
RUN mkdir -p /app/outputs /app/logs /app/models /app/reports

# Set permissions
RUN chmod +x src/fraud_detector.py

# Expose port for potential web interface
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import src.fraud_detector; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "src.fraud_detector", "--help"]