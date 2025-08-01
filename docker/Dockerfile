FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Python + pip
RUN apt-get update && apt-get install -y python3.10 python3-pip git

# Torch + CUDA support
RUN pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Science stack
RUN pip install rdkit-pypi matplotlib pandas seaborn scikit-learn \
    jupyterlab tqdm hydra-core protobuf wandb

# GNN + Diffusion
RUN pip install torch_geometric torch-scatter torch-sparse \
    dgl openbabel prolif codecarbon

# Local project files
WORKDIR /app
COPY . .

# Default entrypoint
CMD [ "bash" ] 