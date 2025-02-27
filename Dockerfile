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

# Copy model fetcher and run it
COPY builder/model_fetcher.py /model_fetcher.py
RUN python3.11 /model_fetcher.py && rm /model_fetcher.py

# Copy the source code
ADD src .

# Set workdir for execution
WORKDIR /src

# Start the handler
CMD ["python3.11", "-u", "handler.py"]