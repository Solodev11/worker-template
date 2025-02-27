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

# Create cache directory
mkdir -p /root/.cache/huggingface

# Download models using the fetcher script
python3.11 /model_fetcher.py

# Clean up
apt-get autoremove -y && \
apt-get clean -y && \
rm -rf /var/lib/apt/lists/*

echo "Setup completed successfully"