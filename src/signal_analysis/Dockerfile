# Use a lightweight Python base image
FROM python:3.9-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (optional, but good practice)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python libraries
# We include 'pyarrow' which Polars uses for Parquet operations
RUN pip install --no-cache-dir streamlit polars plotly pyarrow

# Copy your scripts into the container
COPY convert_data.py dashboard.py /app/

# Expose the default Streamlit port
EXPOSE 8501

# Default command: Run the dashboard
# We use the list format for CMD to properly handle signals
CMD ["streamlit", "run", "dashboard.py", "--server.address=0.0.0.0"]
