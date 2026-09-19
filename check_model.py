import os

model_dir = r"E:\AIGC_Project\models\wd-1-5-beta2"

files = [
    "model_index.json",
    r"unet\config.json",
    r"unet\diffusion_pytorch_model.safetensors",
    r"vae\config.json",
    r"vae\diffusion_pytorch_model.safetensors",
    r"text_encoder\config.json",
    r"tokenizer\tokenizer_config.json",
    r"scheduler\scheduler_config.json",
]

print("===== 检查本地模型 =====")

for file in files:
    path = os.path.join(model_dir, file)

    if os.path.exists(path):
        size_gb = os.path.getsize(path) / 1024**3
        print(f"OK   {file:<50} {size_gb:.3f} GB")
    else:
        print(f"缺失  {file}")