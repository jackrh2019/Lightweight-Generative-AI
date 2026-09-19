import os
import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

# =========================
# 1. 基础配置
# =========================
# 本地模型路径
model_id = r"E:\AIGC_Project\models\wd-1-5-beta2"

# 输出目录
output_dir = r"E:\AIGC_Project\outputs\anime_fullbody_v2"
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

    # 两个一起关闭
    safety_checker=None,
    feature_extractor=None,

    requires_safety_checker=False
)

# 更适合动漫图的 scheduler
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# 8GB 显存推荐设置
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()

# =========================
# 3. Prompt
# =========================
prompt = (
    "masterpiece, best quality, "
    "1woman, solo, adult woman, "

    # 场景提前，不要放最后
    "beach, tropical beach, ocean, blue sea, sand, "
    "palm trees, sunset, orange sky, summer, "

    # 构图
    "full body, standing on the beach, full body visible, feet visible, "
    "wide shot, environmental composition, "

    # 人物
    "beautiful anime woman, blonde hair, long wavy hair, blue eyes, "
    "seductive expression, soft smile, "
    "curvy body, slim waist, long legs, "

    # 服装
    "stylish bikini, beachwear, "

    # 质量
    "detailed face, detailed eyes, detailed hair, "
    "japanese anime style, clean anime illustration, "
    "warm sunset lighting"
)

negative_prompt = (
    "worst quality, low quality, blurry, "
    "multiple people, duplicate, extra person, "
    "extra arms, extra legs, extra fingers, bad hands, bad feet, "
    "cropped, cut off, bad anatomy, deformed body, "

    # 防止变成室内/纯背景
    "indoors, room, bedroom, studio background, "
    "plain background, white background, city, street, "

    "photorealistic, realistic"
)

# =========================
# 4. 参数设置
# =========================
# 第一次先只跑一个 seed，稳定后再扩展

#seeds = [42]
seeds = [42,77,232,2314,4135,2182]

# 全身图推荐竖图
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
