import io
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import asyncio

app = FastAPI()
gpu_lock = asyncio.Lock()

# Load SDXL img2img pipeline
pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    safety_checker=None  # Disable for personal testing
)
pipeline.to("cuda")

# Optional memory-efficient attention
try:
    pipeline.enable_xformers_memory_efficient_attention()
except Exception:
    print("xformers not available, running without it")

pipeline.enable_vae_tiling()  # saves GPU memory

@app.post("/edit")
async def edit_image(
    prompt: str = Form(...),
    image: UploadFile = File(...)
):
    async with gpu_lock:
        try:
            # Load image
            img_data = await image.read()
            init_image = Image.open(io.BytesIO(img_data)).convert("RGB")

            # Resize to SDXL default
            init_image = init_image.resize((1024, 1024))

            # Generate edited image
            with torch.inference_mode():
                result = pipeline(
                    prompt=prompt,
                    image=init_image,
                    strength=0.75,
                    guidance_scale=7.5,
                    num_inference_steps=25,
                    return_dict=True
                ).images[0]

            # Return PNG
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/png")

        except Exception as e:
            print("Error during /edit:", e)
            return {"error": str(e)}
