FROM python:3.11-slim

# 1. Install System Dependencies
# Nautilus and Numba often require build tools and Rust
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    llvm \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Rust (Required for Nautilus Trader)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# 3. Upgrade Pip
RUN pip install --upgrade pip

# 4. Install Python Dependencies
# We combine your requirements.txt reference with the missing b.py libs
# Added: numba, nautilus-trader, matplotlib, filelock, requests
RUN pip install --no-cache-dir \
    nautilus-trader \
    numba \
    pandas \
    numpy \
    scipy \
    "ray[tune]" \
    matplotlib \
    filelock \
    requests \
    hyperopt \
    dash \
    plotly \
    pyarrow \
    polars

# 5. Set working directory
WORKDIR /app

# 6. Default command (Optional, can be overridden)
CMD ["python", "b.py"]
