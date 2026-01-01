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
pipeline.enable_xformers_memory_efficient_attention()

@app.post("/edit")
async def edit_image(
    prompt: str = Form(...),
    image: UploadFile = File(...)
):
    # Read and preprocess image
    img_data = await image.read()
    init_image = Image.open(io.BytesIO(img_data)).convert("RGB")
    init_image = init_image.resize((1024, 1024))

    with torch.no_grad():
        result = pipeline(
            prompt=prompt,
            image=init_image,
            strength=0.75,
            guidance_scale=7.5,
            num_inference_steps=30
        ).images[0]

    # Return as PNG
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
