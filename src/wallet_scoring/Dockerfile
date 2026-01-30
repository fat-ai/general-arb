# Use a lightweight Python 3.10 base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (basic tools potentially needed for compiling libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Install required Python libraries
# pyarrow is included because Polars uses it for efficient interaction with Pandas
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    requests \
    polars \
    pyarrow

COPY wallet_scoring.py .

RUN echo "PYTHONUNBUFFERED=1" >> /etc/environment && \
    echo "0 6 * * * root /usr/local/bin/python /app/wallet_scoring.py >> /proc/1/fd/1 2>&1" > /etc/cron.d/fetcher-cron

# Apply the cron job
RUN chmod 0644 /etc/cron.d/fetcher-cron
RUN crontab /etc/cron.d/fetcher-cron

# 6. Create data directory (for volume mapping)
RUN mkdir -p /app/data

# 7. Start Cron in Foreground
CMD ["cron", "-f"]
