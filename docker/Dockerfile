# VANGUARD Docker Environment
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

LABEL maintainer="Daniel Barreras <daniel.barreras@research.org>"
LABEL description="VANGUARD: Bayesian MRI Harmonization Environment by Daniel Barreras"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /workspace/vanguard

# Copy requirements and install Python dependencies
COPY requirements/ requirements/
RUN pip install --no-cache-dir -r requirements/base.txt
RUN pip install --no-cache-dir -r requirements/training.txt

# Copy source code
COPY src/ src/
COPY setup.py pyproject.toml ./
COPY configs/ configs/

# Install package in development mode
RUN pip install -e .

# Set up jupyter lab
RUN pip install jupyterlab
EXPOSE 8888

# Default command
CMD ["bash"]
