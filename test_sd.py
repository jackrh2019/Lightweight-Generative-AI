import torch
from diffusers import StableDiffusionPipeline

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# 1. 加载模型
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    dtype=torch.float16,
    use_safetensors=True
)

# 2. 8GB 显存：开启 CPU offload
pipe.enable_model_cpu_offload()

# 3. VAE slicing
pipe.vae.enable_slicing()

prompt = (
    "a beautiful futuristic city at night, "
    "cinematic lighting, highly detailed"
)

# 4. 生成图片
image = pipe(
    prompt=prompt,
    height=512,
    width=512,
    num_inference_steps=20,
    guidance_scale=7.5
).images[0]

# 5. 保存
image.save("first_sd_image.png")

print("生成完成：irst_sd_image.png")