#!/usr/bin/env python
import runpod
import torch
import os
from diffusers import AutoPipelineForText2Image
from typing import Dict
import base64
import io

# Input validation schema
INPUT_SCHEMA = {
    "prompt": {
        "type": str,
        "required": True
    },
    "megapixels": {
        "type": int,
        "default": 1
    },
    "aspect_width": {
        "type": int,
        "default": 2
    },
    "aspect_height": {
        "type": int,
        "default": 3
    },
    "num_inference_steps": {
        "type": int,
        "default": 50
    },
    "guidance_scale": {
        "type": float,
        "default": 7.0
    },
    "joint_attention_scale": {
        "type": float,
        "default": 0.65
    }
}

def handler(event) -> Dict:
    """
    Handler function that generates images based on text prompts
    """
    try:
        # Get input
        input_data = event.get("input", {})
        if not input_data.get("prompt"):
            return {"status": "error", "message": "Prompt is required"}

        # Load pipeline and LoRA weights
        pipeline = AutoPipelineForText2Image.from_pretrained(
            'black-forest-labs/FLUX.1-dev',
            torch_dtype=torch.bfloat16,
            token=os.environ.get("HUGGING_FACE_HUB_TOKEN")
        ).to('cuda')

        pipeline.load_lora_weights(
            'soloai1/fluxtrain2',
            weight_name='my_first_flux_lora_v1_000003500.safetensors',
            token=os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )

        # Calculate dimensions
        megapixels = input_data.get("megapixels", 1)
        aspect_width = input_data.get("aspect_width", 2)
        aspect_height = input_data.get("aspect_height", 3)
        total_pixels = megapixels * 1000000
        width = int((total_pixels * aspect_width / aspect_height) ** 0.5)
        height = int(width * aspect_height / aspect_width)

        # Generate image
        output = pipeline(
            prompt=input_data["prompt"],
            num_inference_steps=input_data.get("num_inference_steps", 50),
            guidance_scale=input_data.get("guidance_scale", 7.0),
            height=height,
            width=width,
            joint_attention_kwargs={"scale": input_data.get("joint_attention_scale", 0.65)}
        )

        # Convert to base64
        buffer = io.BytesIO()
        output.images[0].save(buffer, format="JPEG", quality=100)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return {
            "status": "success",
            "image": image_base64
        }
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler,
        "schema": INPUT_SCHEMA
    })
