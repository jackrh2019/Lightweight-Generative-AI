import torch
from diffusers import StableDiffusionPipeline

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    dtype=torch.float16,
    use_safetensors=True
)

print("\n========== Stable Diffusion 结构 ==========\n")

print("1. Tokenizer:")
print(type(pipe.tokenizer))

print("\n2. Text Encoder:")
print(type(pipe.text_encoder))

print("\n3. U-Net:")
print(type(pipe.unet))

print("\n4. VAE:")
print(type(pipe.vae))

print("\n5. Scheduler:")
print(type(pipe.scheduler))