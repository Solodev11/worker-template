# Use the RunPod base image with CUDA support
FROM runpod/base:0.4.0-cuda11.8.0

WORKDIR /

# Environment variables (Hugging Face token and cache location)
ENV HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
ENV HUGGINGFACE_HUB_CACHE="/root/.cache/huggingface"

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Copy the source code
ADD src .

# Optional: Login to Hugging Face during build if token is provided
RUN if [ -n "${HUGGING_FACE_HUB_TOKEN}" ]; then \
        python3.11 -c "from huggingface_hub.commands.user import login; login('${HUGGING_FACE_HUB_TOKEN}')" ; \
    fi

# Set workdir for execution
WORKDIR /src

# Start the handler
CMD ["python3.11", "-u", "handler.py"]