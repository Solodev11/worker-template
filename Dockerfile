# Base image -> https://github.com/runpod/containers/blob/main/official-templates/base/Dockerfile
# DockerHub -> https://hub.docker.com/r/runpod/base/tags
FROM runpod/base:0.4.0-cuda11.8.0@sha256:4d3d4b3c5a5c2b3a5a5c3b2a5a4d2b3a2b3c5a3b2a5d2b3a3b4c3d3b5c3d4a3

WORKDIR /

# The base image comes with many system dependencies pre-installed to help you get started quickly.
# Please refer to the base image's Dockerfile for more information before adding additional dependencies.
# IMPORTANT: The base image overrides the default huggingface cache location.

# Environment variable will be set by RunPod
ENV HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
ENV HUGGINGFACE_HUB_CACHE="/root/.cache/huggingface"

# Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Add src files
ADD src /src

# Set working directory
WORKDIR /src

CMD ["python3.11", "-u", "handler.py"]