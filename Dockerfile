FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install uv for faster package management
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen

# Copy the rest of the application
COPY . .

# Install the package in development mode
RUN uv pip install -e .

# Create directory for config
RUN mkdir -p /root/.arbor

# Expose port
EXPOSE 7453

# Set default command
CMD ["python", "-m", "arbor.cli", "serve"]