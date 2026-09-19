import os
import torch
from diffusers import StableDiffusionPipeline

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

os.makedirs("outputs/seed", exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    dtype=torch.float16,
    use_safetensors=True,
    local_files_only=True,
    low_cpu_mem_usage=True,
    safety_checker=None,
    requires_safety_checker=False
)

pipe.enable_model_cpu_offload()

prompt = (
    "photorealistic portrait of a beautiful young blonde woman, "
    "blue eyes, wavy hair, fair skin, wearing a bikini, "
    "upper body shot, detailed face, realistic skin texture, "
    "soft natural lighting, high detail, sharp focus"
)

negative_prompt = (
    "low quality, blurry, deformed face, bad anatomy, "
    "extra fingers, extra eyes, distorted eyes, bad hands, "
    "ugly, poorly drawn face, malformed mouth, duplicate"
)

seeds = [42, 100, 2026]

for seed in seeds:
    print(f"\n开始生成 Seed = {seed}")

    generator = torch.Generator(device="cpu").manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=512,
        width=512,
        num_inference_steps=30,
        guidance_scale=8.0,
        generator=generator
    ).images[0]

    save_path = f"outputs/seed/seed_{seed}.png"
    image.save(save_path)

    print(f"生成完成：{save_path}")