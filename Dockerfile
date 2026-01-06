FROM debian:stable-slim

ARG KAGGLE_API_TOKEN
ENV KAGGLE_API_TOKEN=$KAGGLE_API_TOKEN
ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt update && apt install -y \
    curl \
    build-essential

# Set working directory
WORKDIR /app

# uv Installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Python dependencies
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync

# Copy source code
COPY main.py main.py

# Pre-download competition data
RUN uv run python -c "from main import download_data; download_data()"

# Run the application
CMD ["uv", "run", "main.py"]