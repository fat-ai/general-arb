# Use a lightweight Python 3.10 base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (basic tools potentially needed for compiling libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install required Python libraries
# pyarrow is included because Polars uses it for efficient interaction with Pandas
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    requests \
    polars \
    pyarrow

# Copy the python script into the container
# (Replace 'my_script.py' with whatever you named your python file)
COPY wallet_scoring.py .

# Define the command to run the script
CMD ["python", "wallet_scoring.py"]
