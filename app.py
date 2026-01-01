import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import io

app = FastAPI()

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)

pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()

@app.post("/edit")
async def edit_image(
    prompt: str = Form(...),
    images: list[UploadFile] = File(...)
):
    # Load first image as base
    base_image = Image.open(images[0].file).convert("RGB")
    base_image = base_image.resize((1024, 1024))

    # Optional: you can later blend multiple images here
    result = pipe(
        prompt=prompt,
        image=base_image,
        strength=0.7,
        guidance_scale=7.5,
        num_inference_steps=30
    ).images[0]

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
