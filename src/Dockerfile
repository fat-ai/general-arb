# Use an official lightweight Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory inside the container
WORKDIR /app

# 1. Install dependencies
# We copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy ONLY the application source code
# This explicitly grabs only the 5 python files and ignores everything else (like the CSV)
COPY main.py broker.py config.py data.py strategy.py ./

# 3. Create the cache directory (required by config.py)
RUN mkdir -p live_paper_cache

# Run the application
CMD ["python", "main.py"]
