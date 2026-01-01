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
