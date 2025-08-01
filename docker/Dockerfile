FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# CUDA architecture flags for Ampere GPUs
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6"
ENV CUDA_VISIBLE_DEVICES=0

# Python + pip
RUN apt-get update && apt-get install -y python3.10 python3-pip git

# Torch + CUDA support
RUN pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Flash-Attention 2 and PyG
RUN pip install flash-attn==2.3.0 --no-build-isolation && \
    pip install torch-geometric==2.4.0 && \
    pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# Pin key dependencies
RUN pip install rdkit==2023.9.5 && \
    pip install codecarbon==2.3.4 && \
    pip install wandb==0.16.2

# Science stack
RUN pip install matplotlib pandas seaborn scikit-learn \
    jupyterlab tqdm hydra-core protobuf

# GNN + Diffusion
RUN pip install dgl openbabel prolif

# Local project files
WORKDIR /app
COPY . .

# Default entrypoint
CMD [ "bash" ] 