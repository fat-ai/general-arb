# Use a lightweight Python image optimized for data science
FROM python:3.11-slim

# Corrected ENV syntax to avoid legacy warnings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install required libraries
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    pyarrow \
    tqdm

# Copy the processing script
COPY process_data.py .

# Create necessary directories
RUN mkdir -p /app/maps /app/temp_chunks /app/data

# Command to run the script
CMD ["python", "process_data.py"]
