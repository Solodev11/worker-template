#!/bin/bash
set -e

echo "Starting system setup..."

# Update system and install dependencies
apt-get update && apt-get upgrade -y
apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    git \
    wget \
    openssh-server

# Create model directories
mkdir -p /models/unet
mkdir -p /models/vae
mkdir -p /models/loras

# Download model files with authorization
echo "Downloading model files..."
wget --header="Authorization: Bearer ${HUGGING_FACE_HUB_TOKEN}" \
     -O /models/unet/flux1-dev.safetensors \
     https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors

wget --header="Authorization: Bearer ${HUGGING_FACE_HUB_TOKEN}" \
     -O /models/vae/ae.safetensors \
     https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors

# Download LoRA weights
wget -O /models/loras/my_first_flux_lora.safetensors \
     https://huggingface.co/soloai1/fluxtrain2/resolve/main/my_first_flux_lora_v1_000003500.safetensors

echo "Setup completed successfully"