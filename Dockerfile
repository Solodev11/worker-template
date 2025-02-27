# Base image -> https://github.com/runpod/containers/blob/main/official-templates/base/Dockerfile
# DockerHub -> https://hub.docker.com/r/runpod/base/tags
FROM runpod/base:0.4.0-cuda11.8.0

# The base image comes with many system dependencies pre-installed to help you get started quickly.
# Please refer to the base image's Dockerfile for more information before adding additional dependencies.
# IMPORTANT: The base image overrides the default huggingface cache location.

# Environment variable will be set by RunPod
ENV HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}

# Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Add src files
ADD src .

# Login to HuggingFace and download models during build
RUN if [ -n "${HUGGING_FACE_HUB_TOKEN}" ]; then \
        huggingface-cli login --token ${HUGGING_FACE_HUB_TOKEN} && \
        python3.11 -c " \
        from diffusers import AutoPipelineForText2Image; \
        import torch; \
        pipeline = AutoPipelineForText2Image.from_pretrained('black-forest-labs/FLUX.1-dev', token='${HUGGING_FACE_HUB_TOKEN}'); \
        pipeline.load_lora_weights('soloai1/fluxtrain2', weight_name='my_first_flux_lora_v1_000003500.safetensors', token='${HUGGING_FACE_HUB_TOKEN}') \
        "; \
    fi

CMD python3.11 -u /handler.py
```