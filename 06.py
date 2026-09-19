import os
import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

# =========================
# 1. 基础配置
# =========================
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

output_dir = "outputs/anime_fullbody"
os.makedirs(output_dir, exist_ok=True)

# =========================
# 2. 加载模型
# =========================
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    dtype=torch.float16,
    use_safetensors=True,
    local_files_only=True,
    low_cpu_mem_usage=True,
    safety_checker=None,
    requires_safety_checker=False
)

# 更适合风格图
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# 8GB 显存模式
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()

# =========================
# 3. Prompt
# =========================
prompt = (
    "masterpiece, best quality, "
    "a single anime girl, solo, one person, "
    "full body, full body shot, standing, centered composition, "
    "beautiful young woman, blonde hair, blue eyes, gentle smile, "
    "slim body, elegant pose, "
    "Sexy cool outfit, "
    "detailed face, detailed eyes, detailed hair, "
    "clean anime illustration, japanese anime style, "
    "soft lighting, sky background"
)

negative_prompt = (
    "worst quality, low quality, lowres, blurry, "
    "multiple people, two girls, duplicate, cloned face, "
    "extra person, extra head, multiple heads, extra face, "
    "bad anatomy, malformed body, deformed body, distorted body, "
    "extra arms, extra legs, extra hands, extra fingers, bad hands, bad feet, "
    "missing limbs, disconnected limbs, cropped, cut off, "
    "photorealistic, realistic, nsfw"
)

# =========================
# 4. 参数设置
# =========================
seeds = [42, 100, 2026]

width = 512
height = 768
num_inference_steps = 30
guidance_scale = 8.0

# =========================
# 5. 生成
# =========================
for seed in seeds:
    print(f"\n开始生成 Seed = {seed}")

    generator = torch.Generator(device="cpu").manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]

    save_path = os.path.join(output_dir, f"fullbody_seed_{seed}.png")
    image.save(save_path)

    print(f"生成完成：{save_path}")

print("\n全部生成完成！")