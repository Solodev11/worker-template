#!/usr/bin/env python3
"""
RunPod | Model Fetcher for FLUX and LoRA weights
"""

import os
import shutil
from pathlib import Path
from diffusers import AutoPipelineForText2Image
import torch

CACHE_DIR = "/root/.cache/huggingface"
MODEL_ID = "black-forest-labs/FLUX.1-dev"
LORA_ID = "soloai1/fluxtrain2"
LORA_WEIGHT = "my_first_flux_lora_v1_000003500.safetensors"

def fetch_models(token: str = None):
    """Downloads and caches the FLUX model and LoRA weights"""
    cache_path = Path(CACHE_DIR)
    if cache_path.exists():
        shutil.rmtree(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading FLUX model from {MODEL_ID}")
    pipeline = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
        token=token
    )

    print(f"Downloading LoRA weights from {LORA_ID}")
    pipeline.load_lora_weights(
        LORA_ID,
        weight_name=LORA_WEIGHT,
        cache_dir=CACHE_DIR,
        token=token
    )

if __name__ == "__main__":
    fetch_models(os.getenv("HUGGING_FACE_HUB_TOKEN"))