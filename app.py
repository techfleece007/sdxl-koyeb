from fastapi import FastAPI, UploadFile, File
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import torch

app = FastAPI()

# Load SDXL Img2Img
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-img2img-1.0",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

@app.post("/img2img")
async def img2img(file: UploadFile = File(...), prompt: str = "A fantasy landscape"):
    input_image = Image.open(file.file).convert("RGB")
    result = pipe(prompt=prompt, image=input_image, strength=0.75, guidance_scale=7.5)
    output_path = "/app/output.png"
    result.images[0].save(output_path)
    return {"output_file": "output.png"}
