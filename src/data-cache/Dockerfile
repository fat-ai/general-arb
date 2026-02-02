FROM python:3.9-slim

# 1. Install System Dependencies (Cron)
RUN apt-get update && apt-get install -y cron && rm -rf /var/lib/apt/lists/*

# 2. Setup Workspace
WORKDIR /app

# 3. Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Script
COPY download_data.py .

# 5. Setup Cron Job
# We use printf to ensure a newline at the end, otherwise cron may ignore the file.
RUN printf "00 00 * * * /usr/local/bin/python /app/download_data.py >> /proc/1/fd/1 2>&1\n" > /etc/cron.d/fetcher-cron

# Apply the cron job
RUN chmod 0644 /etc/cron.d/fetcher-cron && crontab /etc/cron.d/fetcher-cron

# 6. Create data directory
RUN mkdir -p /app/data

# 7. Start Cron in Foreground
CMD ["cron", "-f"]
