# Use the RunPod base image with CUDA support
FROM runpod/base:0.4.0-cuda11.8.0

WORKDIR /

# Environment variables (Hugging Face token and cache location)
ENV HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Create model directories and download models
RUN mkdir -p /models/unet /models/vae /models/loras && \
    wget --header="Authorization: Bearer ${HUGGING_FACE_HUB_TOKEN}" \
        -O /models/unet/flux1-dev.safetensors \
        https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors && \
    wget --header="Authorization: Bearer ${HUGGING_FACE_HUB_TOKEN}" \
        -O /models/vae/ae.safetensors \
        https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors && \
    wget -O /models/loras/my_first_flux_lora.safetensors \
        https://huggingface.co/soloai1/fluxtrain2/resolve/main/my_first_flux_lora_v1_000003500.safetensors

# Copy the source code
ADD src .

# Set workdir for execution
WORKDIR /src

# Start the handler
CMD ["python3.11", "-u", "handler.py"]