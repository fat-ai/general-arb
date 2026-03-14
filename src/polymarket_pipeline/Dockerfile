FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cron \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install all required Python libraries
RUN pip install --no-cache-dir \
    requests \
    polars \
    pandas \
    numpy \
    scipy \
    statsmodels \
    pyarrow

# Copy your scripts into the container
COPY download_data.py .
COPY wallet_scoring.py .
COPY calibrate_fresh_wallets.py .
COPY run_pipeline.sh .

# Make the bash script executable
RUN chmod +x run_pipeline.sh

# Set up the single cron job to run at midnight every day
RUN echo "0 0 * * * root /app/run_pipeline.sh >> /proc/1/fd/1 2>&1" > /etc/cron.d/pipeline-cron
RUN chmod 0644 /etc/cron.d/pipeline-cron
RUN crontab /etc/cron.d/pipeline-cron

RUN mkdir -p /app/data

CMD ["cron", "-f"]
