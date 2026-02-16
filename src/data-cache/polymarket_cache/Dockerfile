# Use a lightweight Python image optimized for data science
FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files 
# and to ensure output is sent straight to terminal (unbuffered)
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies needed for some python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first to leverage Docker cache
# We include the libraries mentioned in your script
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    pyarrow \
    fastparquet \
    tqdm

# Copy your processing script and the data directory
# Note: We expect the CSV and Parquet to be mounted via volumes 
# rather than copied, to keep the image small.
COPY process_data.py .

# Create directories for maps and temp chunks
RUN mkdir -p /app/maps /app/temp_chunks /app/data

# Command to run the script
# We use the final version filenames as defaults
CMD ["python", "process_data.py"]
