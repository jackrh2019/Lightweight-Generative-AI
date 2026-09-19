import os
import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

# =========================
# 1. 基础配置
# =========================
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# 输出文件夹
output_dir = "outputs/anime_portrait"
os.makedirs(output_dir, exist_ok=True)

# =========================
# 2. 加载模型
# =========================
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    dtype=torch.float16,
    use_safetensors=True,
    local_files_only=True,          # 只使用本地已缓存模型
    low_cpu_mem_usage=True,         # 降低加载时 CPU 内存峰值
    safety_checker=None,            # 不加载 safety checker
    requires_safety_checker=False
)

# 换成更适合风格图的 scheduler
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# 8GB 显存建议：开启 CPU offload
pipe.enable_model_cpu_offload()

# 可选：VAE slicing（进一步降低内存压力）
pipe.vae.enable_slicing()

# =========================
# 3. Prompt 设计
# =========================
prompt = (
    "a single anime girl, solo, one person, "
    "centered portrait, head and shoulders, "
    "beautiful young woman, long purple hair, blue eyes, gentle smile, "
    "detailed eyes, detailed face, detailed hair, "
    "soft lighting, blue sky background, "
    "japanese anime style, clean anime illustration"
)

negative_prompt = (
    "multiple people, two girls, twins, duplicate, cloned face, "
    "extra person, extra head, extra face, extra body, "
    "multiple heads, two heads, disconnected body, "
    "bad anatomy, deformed, distorted face, bad eyes, "
    "extra limbs, extra arms, extra fingers, bad hands, "
    "blurry, low quality, lowres, photorealistic, realistic"
)

# =========================
# 4. 生成设置
# =========================
# 你可以改成只要一个 seed，比如 [42]
seeds = [42, 100, 2026]

# 建议先固定这些参数
height = 768
width = 512
num_inference_steps = 30
guidance_scale = 8.0

# =========================
# 5. 开始生成
# =========================
for seed in seeds:
    print(f"\n开始生成 Seed = {seed}")

    # 固定随机种子，保证结果可复现
    generator = torch.Generator(device="cpu").manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]

    save_path = os.path.join(output_dir, f"anime_seed_{seed}.png")
    image.save(save_path)

    print(f"生成完成：{save_path}")

print("\n全部生成完成！")