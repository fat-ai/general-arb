FROM python:3.9-slim

# 1. Install System Dependencies (Cron)
RUN apt-get update && apt-get install -y cron && rm -rf /var/lib/apt/lists/*

# 2. Setup Workspace
WORKDIR /app

# 3. Install Python Dependencies
# (Assuming you have these. If not, create requirements.txt with: pandas requests numpy pyarrow)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Script
COPY download_data.py .

# 5. Setup Cron Job
# This line creates the cron schedule file
# 0 0 * * * = Run at midnight every day
# >> /proc/1/fd/1 2>&1 = Redirect output to Docker logs so you can see it via 'docker logs'
RUN echo "0 0 * * * /usr/local/bin/python /app/download_data.py >> /proc/1/fd/1 2>&1" > /etc/cron.d/fetcher-cron

# Apply the cron job
RUN chmod 0644 /etc/cron.d/fetcher-cron
RUN crontab /etc/cron.d/fetcher-cron

# 6. Create data directory (for volume mapping)
RUN mkdir -p /app/data

# 7. Start Cron in Foreground
CMD ["cron", "-f"]
