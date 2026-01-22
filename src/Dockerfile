# Use an official lightweight Python image
FROM python:3.11-slim

# Set environment variables
# PYTHONUNBUFFERED=1 ensures logs are printed immediately to the console
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory inside the container
WORKDIR /app

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create the cache directory defined in config.py to ensure permissions are correct
RUN mkdir -p live_paper_cache

# Run the application
CMD ["python", "main.py"]
