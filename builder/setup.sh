#!/bin/bash
set -e  # Stop script on error

echo "Starting system setup..."

# Update system
apt-get update && apt-get upgrade -y

# Install system dependencies
apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    git \
    openssh-server

# Cache HuggingFace models during build
echo "Caching HuggingFace models..."
python3.11 -c "
from diffusers import AutoPipelineForText2Image
import torch

# Initialize and cache the models
pipeline = AutoPipelineForText2Image.from_pretrained(
    'black-forest-labs/FLUX.1-dev',
    torch_dtype=torch.bfloat16
)

# Cache LoRA weights
pipeline.load_lora_weights(
    'soloai1/fluxtrain2',
    weight_name='my_first_flux_lora_v1_000003500.safetensors'
)
"

# Clean up
apt-get autoremove -y && \
apt-get clean -y && \
rm -rf /var/lib/apt/lists/*

echo "Setup completed successfully"