import io
import torch
import asyncio
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image

app = FastAPI()

# Lock to prevent GPU overload
gpu_lock = asyncio.Lock()

# Load SDXL pipeline (no safety checker)
pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    safety_checker=None,
)
pipeline.to("cuda")

# Memory optimization
try:
    pipeline.enable_xformers_memory_efficient_attention()
except Exception:
    print("xformers not installed, running without it")

pipeline.enable_vae_tiling()

@app.post("/edit")
async def edit_image(prompt: str = Form(...), image: UploadFile = File(...)):
    async with gpu_lock:
        try:
            img_data = await image.read()
            init_image = Image.open(io.BytesIO(img_data)).convert("RGB")
            if init_image.size != (1024, 1024):
                init_image = init_image.resize((1024, 1024), Image.LANCZOS)

            with torch.inference_mode():
                result = pipeline(
                    prompt=prompt,
                    image=init_image,
                    strength=0.75,
                    guidance_scale=7.5,
                    num_inference_steps=25,
                ).images[0]

            buf = io.BytesIO()
            result.save(buf, format="PNG")
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/png")

        except Exception as e:
            return {"error": str(e)}
