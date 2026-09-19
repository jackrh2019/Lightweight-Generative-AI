from huggingface_hub import snapshot_download

repo_id = "waifu-diffusion/wd-1-5-beta2"
local_dir = r"E:\AIGC_Project\models\wd-1-5-beta2"

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,

    # 只下载 Diffusers 推理需要的东西
    allow_patterns=[
        "model_index.json",
        "scheduler/*",
        "tokenizer/*",
        "text_encoder/*",
        "unet/*",
        "vae/*",
        "feature_extractor/*",
    ],

    # 网络不稳定时少开几个并发连接
    max_workers=1
)

print("模型下载完成")