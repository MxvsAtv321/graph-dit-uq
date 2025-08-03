FROM python:3.10-slim

# CUDA architecture flags for Ampere GPUs
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6"
ENV CUDA_VISIBLE_DEVICES=0

# System dependencies
RUN apt-get update && apt-get install -y git curl

# Torch + CPU support (for now)
RUN pip install torch==2.0.1 torchvision torchaudio

# Install basic packages (skip PyG for now)
RUN pip install fastapi uvicorn requests

# Pin key dependencies
RUN pip install rdkit==2023.9.5 && \
    pip install wandb==0.16.2

# Science stack
RUN pip install matplotlib pandas seaborn scikit-learn \
    tqdm protobuf

# Local project files
WORKDIR /app
COPY . .

# Default entrypoint
CMD [ "bash" ] 