import io
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

app = FastAPI()

# Load SDXL img2img pipeline
pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True
)

pipeline.to("cuda")

# Safe optimizations (DO NOT crash if unavailable)
pipeline.enable_attention_slicing()

try:
    pipeline.enable_xformers_memory_efficient_attention()
    print("xformers enabled")
except Exception:
    print("xformers not available, running without it")

@app.post("/edit")
async def edit_image(
    prompt: str = Form(...),
    image: UploadFile = File(...)
):
    img_data = await image.read()
    init_image = Image.open(io.BytesIO(img_data)).convert("RGB")
    init_image = init_image.resize((1024, 1024))

    with torch.inference_mode():
        result = pipeline(
            prompt=prompt,
            image=init_image,
            strength=0.75,
            guidance_scale=7.5,
            num_inference_steps=30
        ).images[0]

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
