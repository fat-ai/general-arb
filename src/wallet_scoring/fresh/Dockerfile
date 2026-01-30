FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr (so you see logs instantly)
ENV PYTHONUNBUFFERED=1

# Install system dependencies (sometimes needed for pyarrow/numpy/scipy)
# usually not strictly necessary for binary wheels, but good for safety
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python libraries
# We include pyarrow because Polars/Pandas Parquet support relies on it
RUN pip install --no-cache-dir \
    polars \
    pandas \
    numpy \
    scipy \
    statsmodels \
    pyarrow

COPY calibrate_fresh_wallets.py .

RUN echo "PYTHONUNBUFFERED=1" >> /etc/environment && \
    echo "0 9 * * * root /usr/local/bin/python /app/calibrate_fresh_wallets.py >> /proc/1/fd/1 2>&1" > /etc/cron.d/fetcher-cron

# Apply the cron job
RUN chmod 0644 /etc/cron.d/fetcher-cron
RUN crontab /etc/cron.d/fetcher-cron

# 6. Create data directory (for volume mapping)
RUN mkdir -p /app/data

# 7. Start Cron in Foreground
CMD ["cron", "-f"]
