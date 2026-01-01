import io
import torch
import asyncio
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image

app = FastAPI()

# GPU lock to prevent concurrent VRAM overload
gpu_lock = asyncio.Lock()

# Load SDXL Img2Img pipeline
pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",           # saves ~6GB VRAM
    use_safetensors=True,
    safety_checker=None
)
pipeline.to("cuda")

# Enable memory improvements if available
try:
    pipeline.enable_xformers_memory_efficient_attention()
except Exception:
    print("xformers not available, running without memory-efficient attention")

pipeline.enable_vae_tiling()  # reduces VRAM usage

@app.post("/edit")
async def edit_image(
    prompt: str = Form(...),
    image: UploadFile = File(...)
):
    async with gpu_lock:
        try:
            # Read input image
            img_data = await image.read()
            init_image = Image.open(io.BytesIO(img_data)).convert("RGB")

            # Resize to 1024x1024 if needed
            if init_image.size != (1024, 1024):
                init_image = init_image.resize((1024, 1024), Image.LANCZOS)

            # Run SDXL Img2Img
            with torch.inference_mode():
                result = pipeline(
                    prompt=prompt,
                    image=init_image,
                    strength=0.75,
                    guidance_scale=7.5,
                    num_inference_steps=25
                ).images[0]

            # Return as PNG
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/png")

        except Exception as e:
            print(f"Error during /edit: {e}")
            return {"error": str(e)}
