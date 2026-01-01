import io
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

app = FastAPI()

# Load SDXL img2img pipeline (no safety checker)
pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    safety_checker=None  # disable safety filtering for personal use
)
pipeline.to("cuda")

# Optional: enable memory-efficient attention if xformers is installed
try:
    pipeline.enable_xformers_memory_efficient_attention()
except Exception:
    print("xformers not installed, running without memory-efficient attention")

@app.post("/edit")
async def edit_image(
    prompt: str = Form(...),
    image: UploadFile = File(...)
):
    # Read uploaded image
    img_data = await image.read()
    init_image = Image.open(io.BytesIO(img_data)).convert("RGB")
    
    # Resize to 1024x1024 only if smaller; else keep original
    init_image = init_image.resize((1024, 1024)) if init_image.size != (1024, 1024) else init_image

    # Run SDXL
    with torch.no_grad():
        result_image = pipeline(
            prompt=prompt,
            image=init_image,
            strength=0.75,
            guidance_scale=7.5,
            num_inference_steps=30
        ).images[0]

    # Return as PNG
    buf = io.BytesIO()
    result_image.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")



import io
import torch
import asyncio
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from diffusers import StableDiffusionXLImg2ImgPipeline  # FIXED CLASS
from PIL import Image

app = FastAPI()

# Global lock to prevent GPU overload (Crucial for SDXL's high VRAM usage)
gpu_lock = asyncio.Lock()

# Load SDXL img2img pipeline
# Note: variant="fp16" is highly recommended to save ~6GB of VRAM
pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",  # 2026 standard for consumer GPUs
    use_safetensors=True,
    safety_checker=None
)
pipeline.to("cuda")

# Memory improvements
try:
    pipeline.enable_xformers_memory_efficient_attention()
except Exception:
    print("xformers not available, running without it")

# Reduces memory usage significantly for 1024x1024 images
pipeline.enable_vae_tiling() 

@app.post("/edit")
async def edit_image(
    prompt: str = Form(...),
    image: UploadFile = File(...)
):
    async with gpu_lock:
        try:
            # Read image
            img_data = await image.read()
            init_image = Image.open(io.BytesIO(img_data)).convert("RGB")

            # SDXL is optimized for 1024x1024
            if init_image.size != (1024, 1024):
                init_image = init_image.resize((1024, 1024), Image.LANCZOS)

            # Generate image
            with torch.inference_mode():
                result = pipeline(
                    prompt=prompt,
                    image=init_image,
                    strength=0.75,
                    guidance_scale=7.5,
                    num_inference_steps=25
                ).images[0]

            # Return result
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/png")

        except Exception as e:
            print(f"Error during /edit: {e}")
            return {"error": str(e)}
