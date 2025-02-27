#!/usr/bin/env python
import runpod
import torch
from diffusers import AutoPipelineForText2Image
from typing import Dict

# Input validation schema
INPUT_SCHEMA = {
    "prompt": {
        "type": str,
        "required": True
    },
    "megapixels": {
        "type": int,
        "default": 1,
        "min": 1,
        "max": 4
    },
    "aspect_width": {
        "type": int,
        "default": 2,
        "min": 1
    },
    "aspect_height": {
        "type": int,
        "default": 3,
        "min": 1
    },
    "num_inference_steps": {
        "type": int,
        "default": 50,
        "min": 1,
        "max": 100
    },
    "guidance_scale": {
        "type": float,
        "default": 7.0,
        "min": 1.0,
        "max": 20.0
    },
    "joint_attention_scale": {  # Added new parameter
        "type": float,
        "default": 0.65,
        "min": 0.0,
        "max": 1.0
    }
}

# Initialize the pipeline globally for reuse
@runpod.init({"schema": INPUT_SCHEMA})
def init_pipeline():
    try:
        pipeline = AutoPipelineForText2Image.from_pretrained(
            'black-forest-labs/FLUX.1-dev',
            torch_dtype=torch.bfloat16
        ).to('cuda')
        
        pipeline.load_lora_weights(
            'soloai1/fluxtrain2',
            weight_name='my_first_flux_lora_v1_000003500.safetensors'
        )
        return pipeline
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model: {str(e)}")

# Global pipeline instance
pipeline = init_pipeline()

@runpod.runpod.handler(input_schema=INPUT_SCHEMA)
def handler(event) -> Dict:
    try:
        input_data = event["input"]
        
        # Calculate dimensions
        total_pixels = input_data["megapixels"] * 1000000
        width = int((total_pixels * input_data["aspect_width"] / input_data["aspect_height"]) ** 0.5)
        height = int(width * input_data["aspect_height"] / input_data["aspect_width"])

        # Set joint attention kwargs
        joint_attention_kwargs = {"scale": input_data.get("joint_attention_scale", 0.65)}

        # Generate image
        output = pipeline(
            input_data["prompt"],
            num_inference_steps=input_data["num_inference_steps"],
            guidance_scale=input_data["guidance_scale"],
            height=height,
            width=width,
            joint_attention_kwargs=joint_attention_kwargs  # Added parameter
        )

        # Convert to base64
        import io
        import base64
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
