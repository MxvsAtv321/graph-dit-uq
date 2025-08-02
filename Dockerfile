FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for molecular AI (without CUDA dependencies)
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    requests \
    pyyaml \
    rdkit-pypi

# Create working directory
WORKDIR /app

# Copy the source code
COPY src/ /app/src/
COPY requirements.txt /app/

# Install only the basic requirements (skip problematic ones)
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    torch-geometric>=2.4.0 \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    requests \
    pyyaml

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python"] 