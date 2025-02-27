# Base image -> https://github.com/runpod/containers/blob/main/official-templates/base/Dockerfile
# DockerHub -> https://hub.docker.com/r/runpod/base/tags
FROM runpod/base:0.4.0-cuda11.8.0

WORKDIR /

# The base image comes with many system dependencies pre-installed.
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
ADD src .

# Login to HuggingFace
RUN if [ -n "${HUGGING_FACE_HUB_TOKEN}" ]; then \
        python3.11 -c "from huggingface_hub.commands.user import login; login('${HUGGING_FACE_HUB_TOKEN}')" \
    fi

# Set working directory
WORKDIR /src

CMD ["python3.11", "-u", "handler.py"]