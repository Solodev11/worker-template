#!/usr/bin/env python
import runpod
import torch
from diffusers import AutoPipelineForText2Image
from typing import Dict
import base64
import io

# Global pipeline instance
pipeline = None

def init_pipeline():
    """Initialize the pipeline on first use"""
    global pipeline
    if pipeline is None:
        try:
            pipeline = AutoPipelineForText2Image.from_pretrained(
                'black-forest-labs/FLUX.1-dev',
                torch_dtype=torch.bfloat16
            ).to('cuda')
            
            pipeline.load_lora_weights(
                'soloai1/fluxtrain2',
                weight_name='my_first_flux_lora_v1_000003500.safetensors'
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")
    return pipeline

def handler(event) -> Dict:
    """
    Handler function that generates images based on text prompts
    """
    try:
        # Get input
        input_data = event.get("input", {})
        if not input_data.get("prompt"):
            return {"status": "error", "message": "Prompt is required"}

        # Initialize pipeline if needed
        global pipeline
        if pipeline is None:
            pipeline = init_pipeline()

        # Set defaults
        megapixels = input_data.get("megapixels", 1)
        aspect_width = input_data.get("aspect_width", 2)
        aspect_height = input_data.get("aspect_height", 3)
        num_inference_steps = input_data.get("num_inference_steps", 50)
        guidance_scale = input_data.get("guidance_scale", 7.0)
        joint_attention_scale = input_data.get("joint_attention_scale", 0.65)

        # Calculate dimensions
        total_pixels = megapixels * 1000000
        width = int((total_pixels * aspect_width / aspect_height) ** 0.5)
        height = int(width * aspect_height / aspect_width)

        # Generate image
        output = pipeline(
            input_data["prompt"],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            joint_attention_kwargs={"scale": joint_attention_scale}
        )

        # Convert to base64
        buffer = io.BytesIO()
        output.images[0].save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return {
            "status": "success",
            "image": image_base64
        }
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
