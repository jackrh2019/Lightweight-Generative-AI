import os
import torch
from diffusers import StableDiffusionPipeline

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

os.makedirs("outputs/seed", exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    dtype=torch.float16,
    use_safetensors=True,
    local_files_only=True,      # 只用已经下载好的本地模型
    low_cpu_mem_usage=True,     # 降低模型加载时的 CPU 内存峰值
    safety_checker=None,        # 少加载一个组件
    requires_safety_checker=False
)

pipe.enable_model_cpu_offload()

prompt = (
    "a beautiful futuristic city at night, "
    "cinematic lighting, highly detailed"
)

seeds = [42, 100, 2026]

for seed in seeds:
    print(f"\n开始生成 Seed = {seed}")

    generator = torch.Generator(device="cpu").manual_seed(seed)

    image = pipe(
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=20,
        guidance_scale=7.5,
        generator=generator
    ).images[0]

    save_path = f"outputs/seed/seed_{seed}.png"
    image.save(save_path)

    print(f"生成完成：{save_path}")