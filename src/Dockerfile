# Use an official lightweight Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory inside the container
WORKDIR /app

RUN apt-get update && apt-get install -y libjemalloc2 && rm -rf /var/lib/apt/lists/*

# Force the container to use jemalloc for all memory allocations instead of glibc
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

# 1. Install dependencies
# We copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy ONLY the application source code
# This explicitly grabs only the 5 python files and ignores everything else (like the CSV)
COPY *.py ./

# 3. Create the cache directory (required by config.py)
RUN mkdir -p live_paper_cache

# Run the application
# Start the Uvicorn web server listening on all interfaces
CMD ["python", "main.py"]
