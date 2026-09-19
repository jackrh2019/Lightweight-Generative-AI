# Lightweight Generative AI Lab

A lightweight hands-on project for learning and experimenting with diffusion-based generative AI on consumer GPUs.

This project currently focuses on:

- Stable Diffusion inference
- Latent Diffusion architecture
- Prompt engineering
- Random seed experiments
- Anime image generation
- Anime-specific diffusion checkpoints
- Memory-efficient inference on consumer GPUs

The project will gradually expand toward:

- LoRA fine-tuning
- ControlNet
- Image-to-Video generation
- Video Diffusion
- Temporal consistency
- Audio-conditioned video generation

The current experiments were developed and tested on an **NVIDIA RTX 3070 Laptop GPU with 8 GB VRAM**.

---

# 1. Project Goals

The goal of this repository is to build a practical understanding of modern generative AI systems through hands-on experiments.

The project follows the learning path:

```text
Stable Diffusion
       ↓
Prompt Engineering
       ↓
Anime-Specific Checkpoints
       ↓
LoRA Fine-tuning
       ↓
Controllable Generation
       ↓
Image-to-Video
       ↓
Video Diffusion
       ↓
Temporal Consistency
       ↓
Audio / Music Conditioned Video Generation
```

The long-term objective is to explore lightweight image and video generation pipelines that can run on consumer GPUs.

---

# 2. Current Experiments

## 2.1 Stable Diffusion 1.5 Inference

Basic text-to-image generation using Stable Diffusion 1.5.

The simplified generation pipeline is:

```text
Prompt
  ↓
Tokenizer
  ↓
CLIP Text Encoder
  ↓
Text Embedding
  ↓
Random Latent Noise
  ↓
U-Net Denoising
  ↓
Scheduler
  ↓
Latent Representation
  ↓
VAE Decoder
  ↓
Generated Image
```

---

## 2.2 Stable Diffusion Architecture Inspection

The project inspects the major modules inside Stable Diffusion.

Example components:

```text
Tokenizer      → CLIPTokenizer
Text Encoder   → CLIPTextModel
U-Net          → UNet2DConditionModel
VAE            → AutoencoderKL
Scheduler      → PNDMScheduler / Euler Scheduler
```

This experiment helps explain how Stable Diffusion performs conditional image generation.

---

## 2.3 Random Seed Experiment

Different random seeds are tested while keeping the following parameters fixed:

- Prompt
- Model
- Number of inference steps
- Guidance scale
- Image resolution

Example:

```text
Seed 42
   ↓
Initial Noise A
   ↓
Diffusion Process
   ↓
Image A

Seed 100
   ↓
Initial Noise B
   ↓
Diffusion Process
   ↓
Image B
```

This demonstrates that the random seed determines the initial latent noise and therefore affects the final generated image.

---

## 2.4 Prompt Engineering

The project currently experiments with:

- Positive prompts
- Negative prompts
- Character descriptions
- Scene descriptions
- Full-body composition
- Portrait generation
- Anime-style generation
- Background control
- Sampling steps
- Guidance scale
- Random seeds

The goal is to understand how different textual conditions affect:

- Image quality
- Character structure
- Composition
- Style
- Semantic alignment
- Background generation

---

## 2.5 Anime Image Generation

In addition to the original Stable Diffusion 1.5 model, this repository experiments with an anime-oriented diffusion checkpoint:

```text
Waifu Diffusion 1.5 Beta 2
```

Current experiments include:

- Anime portraits
- Full-body anime characters
- Character styling
- Beach scenes
- Fashion scenes
- Anime illustration generation

Large model weights are **not included in this Git repository**.

Users should download the required model separately using the included download script.

---

# 3. Hardware and Software Environment

The project was developed and tested with:

```text
Operating System: Windows 11

GPU:
NVIDIA GeForce RTX 3070 Laptop GPU

VRAM:
8 GB

Python:
3.10

PyTorch:
2.5.1

CUDA Runtime:
12.4

Diffusers:
0.40.0

Pillow:
10.4.0
```

Because the GPU has only 8 GB VRAM, memory-efficient inference techniques are used.

For example:

```python
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
```

---

# 4. Project Structure

The repository currently follows approximately this structure:

```text
Lightweight-Generative-AI/
│
├── 02_inspect_pipeline.py
├── 03_seed_experiment.py
├── 04_test.py
├── 05_anime_portrait.py
├── 06.py
├── 07.py
│
├── check_model.py
├── download_wd15.py
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── models/
│   └── wd-1-5-beta2/
│
└── outputs/
```

The following folders are ignored by Git:

```text
models/
outputs/
```

Large model weights and generated images are therefore not uploaded to GitHub by default.

---

# 5. Quick Start

For users who already have **Git and Conda installed**, the basic workflow is:

```bash
git clone https://github.com/jackrh2019/Lightweight-Generative-AI.git

cd Lightweight-Generative-AI

conda create -n aigc_sd python=3.10 -y

conda activate aigc_sd

conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y

pip install -r requirements.txt

python download_wd15.py

python 07.py
```

Generated images will be stored under:

```text
outputs/
```

For detailed installation instructions, continue reading below.

---

# 6. Installation

## Step 1 — Clone the Repository

Clone the project:

```bash
git clone https://github.com/jackrh2019/Lightweight-Generative-AI.git
```

Enter the project directory:

```bash
cd Lightweight-Generative-AI
```

---

## Step 2 — Create a Conda Environment

Create a clean Python 3.10 environment:

```bash
conda create -n aigc_sd python=3.10 -y
```

Activate it:

```bash
conda activate aigc_sd
```

You should now see something similar to:

```text
(aigc_sd)
```

at the beginning of your terminal.

---

## Step 3 — Install PyTorch and CUDA Runtime

This project was tested with:

```text
PyTorch 2.5.1
CUDA Runtime 12.4
```

Install them using Conda:

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

You do **not** need to manually install the complete CUDA Toolkit for this project.

The `pytorch-cuda=12.4` package provides the CUDA runtime required by PyTorch.

---

## Step 4 — Verify GPU Support

Run:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

Expected output should look similar to:

```text
PyTorch: 2.5.1
CUDA: 12.4
CUDA available: True
GPU: NVIDIA GeForce RTX 3070 Laptop GPU
```

The exact GPU name will depend on your hardware.

---

## Step 5 — Install Python Dependencies

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The main libraries used by this project include:

```text
diffusers
transformers
accelerate
safetensors
huggingface-hub
Pillow
```

A recommended `requirements.txt` is:

```text
diffusers==0.40.0
transformers==5.16.0
accelerate==1.14.0
safetensors
huggingface-hub
Pillow==10.4.0
```

PyTorch is intentionally installed separately using Conda so that the correct CUDA-enabled build is used.

---

# 7. Download the Anime Diffusion Model

Large model weights are not stored inside this GitHub repository.

Run:

```bash
python download_wd15.py
```

The model will be downloaded into:

```text
models/wd-1-5-beta2/
```

The directory should approximately contain:

```text
models/
└── wd-1-5-beta2/
    ├── model_index.json
    ├── scheduler/
    ├── tokenizer/
    ├── text_encoder/
    ├── unet/
    ├── vae/
    └── feature_extractor/
```

Important files include:

```text
unet/diffusion_pytorch_model.safetensors
vae/diffusion_pytorch_model.safetensors
```

---

## Model Download Script

The download script uses Hugging Face Hub.

Example:

```python
from huggingface_hub import snapshot_download
import os

repo_id = "waifu-diffusion/wd-1-5-beta2"

local_dir = os.path.join(
    "models",
    "wd-1-5-beta2"
)

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    allow_patterns=[
        "model_index.json",
        "scheduler/*",
        "tokenizer/*",
        "text_encoder/*",
        "unet/*",
        "vae/*",
        "feature_extractor/*",
    ],
    max_workers=2
)

print("Model download completed.")
```

If the download is interrupted, simply run:

```bash
python download_wd15.py
```

again.

Already downloaded files will normally be reused.

---

# 8. Check the Downloaded Model

Run:

```bash
python check_model.py
```

The script can be used to verify that important model components exist locally.

For example:

```text
model_index.json
unet/config.json
unet/diffusion_pytorch_model.safetensors
vae/config.json
vae/diffusion_pytorch_model.safetensors
text_encoder/config.json
tokenizer/tokenizer_config.json
scheduler/scheduler_config.json
```

---

# 9. Running the Experiments

## Inspect Stable Diffusion Architecture

Run:

```bash
python 02_inspect_pipeline.py
```

This displays major Stable Diffusion components such as:

```text
Tokenizer
Text Encoder
U-Net
VAE
Scheduler
```

---

## Random Seed Experiment

Run:

```bash
python 03_seed_experiment.py
```

This compares images generated from different random seeds.

---

## Anime Portrait Generation

Run:

```bash
python 05_anime_portrait.py
```

Generated images will be saved under:

```text
outputs/
```

---

## Full-Body Anime Generation

Run:

```bash
python 07.py
```

This script currently contains experiments with:

- Full-body anime characters
- Anime-specific checkpoints
- Positive prompts
- Negative prompts
- Different random seeds
- Scene generation
- Memory-efficient inference

Generated images are stored under:

```text
outputs/
```

---

# 10. PyCharm Setup

This project can also be run directly inside PyCharm.

After creating the Conda environment, configure PyCharm to use it.

Go to:

```text
Settings
→ Project
→ Python Interpreter
→ Add Interpreter
→ Conda Environment
→ Existing Environment
```

Select the Python executable inside the environment.

For example:

```text
E:\anaconda\envs\aigc_sd\python.exe
```

The exact path depends on where Anaconda or Miniconda is installed.

After configuring the interpreter, the scripts can be run directly from PyCharm.

---

# 11. Example Stable Diffusion Pipeline

A memory-efficient pipeline configuration used in this project looks like:

```python
import torch
from diffusers import StableDiffusionPipeline

model_id = "your-model-path"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    dtype=torch.float16,
    use_safetensors=True,
    local_files_only=True,
    low_cpu_mem_usage=True,
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False
)

pipe.enable_model_cpu_offload()

pipe.vae.enable_slicing()
```

Example generation:

```python
generator = torch.Generator(
    device="cpu"
).manual_seed(42)

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=512,
    height=768,
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=generator
).images[0]

image.save("output.png")
```

---

# 12. Core Concepts

## VAE

The VAE maps images between pixel space and a compressed latent space.

Encoding:

```text
Image
 ↓
VAE Encoder
 ↓
Latent Representation
```

Decoding:

```text
Generated Latent
 ↓
VAE Decoder
 ↓
Image
```

Stable Diffusion performs most of the diffusion process in latent space instead of directly in pixel space.

This significantly reduces computational cost.

---

## U-Net

The U-Net is the main denoising network in Stable Diffusion 1.5.

Simplified process:

```text
Noisy Latent
+
Text Condition
      ↓
     U-Net
      ↓
Noise Prediction
```

The model repeatedly predicts noise during the reverse diffusion process.

---

## CLIP Text Encoder

The text encoder converts text prompts into semantic embeddings.

```text
Prompt
  ↓
Tokenizer
  ↓
Token IDs
  ↓
CLIP Text Encoder
  ↓
Text Embedding
```

The text embedding is used to condition the image generation process.

---

## Scheduler

The scheduler determines how the latent representation is updated during each diffusion step.

A simplified interpretation is:

```text
U-Net
→ predicts the noise

Scheduler
→ determines how the latent should be updated
```

---

## Random Seed

The random seed controls the initial latent noise.

Therefore:

```text
Same Prompt
+
Different Seed
=
Different Generated Image
```

Using the same:

```text
Model
Prompt
Seed
Inference Steps
Guidance Scale
Resolution
```

helps make experiments reproducible.

---

## Guidance Scale

`guidance_scale` controls how strongly the generated image follows the text prompt.

For example:

```python
guidance_scale = 7.5
```

Higher values usually increase prompt adherence, but extremely high values may reduce naturalness or image quality.

---

## Inference Steps

Example:

```python
num_inference_steps = 30
```

This controls how many denoising iterations are performed.

More steps generally increase computation time.

More steps do **not necessarily** guarantee better image quality.

---

# 13. Prompt Length Limitation

Stable Diffusion 1.5 uses a CLIP text encoder with a limited context length.

Very long prompts may be truncated.

If you see a warning similar to:

```text
CLIP can only handle sequences up to 77 tokens
```

part of the prompt is being ignored.

For best results:

- Keep important concepts near the beginning
- Avoid unnecessary repeated adjectives
- Use concise visual descriptions
- Prioritize character, composition, clothing and scene information

Example:

```text
masterpiece, 1woman, solo, full body,
blonde hair, blue eyes,
beach, ocean, sunset,
anime style
```

is often more useful than a very long descriptive sentence.

---

# 14. Memory Optimization

Consumer GPUs may not have enough VRAM to load every model component simultaneously.

This project uses:

```python
pipe.enable_model_cpu_offload()
```

This keeps inactive model components in CPU memory and moves them to the GPU when needed.

VAE slicing can also be enabled:

```python
pipe.vae.enable_slicing()
```

Other memory optimization techniques that may be explored later include:

- FP16 inference
- Gradient checkpointing
- Attention optimization
- LoRA
- Quantization
- Model offloading

---

# 15. Common Issues

## CUDA Is Not Available

Test:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If the result is:

```text
False
```

verify that the CUDA-enabled PyTorch build was installed correctly.

---

## Hugging Face Download Fails

Possible errors include:

```text
SSL CERTIFICATE_VERIFY_FAILED
```

or:

```text
WinError 10054
```

This may be caused by:

- Campus networks
- Corporate networks
- HTTPS inspection
- Proxy configuration
- Firewall configuration
- Certificate-chain issues

Possible solutions include:

- Switch to another network
- Use a mobile hotspot
- Check proxy settings
- Retry the download

Once the model has been fully downloaded locally, inference can use:

```python
local_files_only=True
```

and no Internet connection is required.

---

## Pillow DLL Error

On some Windows + Conda environments, Pillow may produce:

```text
ImportError: DLL load failed while importing _imaging
```

A possible fix is:

```bash
pip install --force-reinstall --no-cache-dir Pillow==10.4.0
```

Then restart the terminal or PyCharm.

---

## GPU Memory Error

If GPU memory is insufficient, enable:

```python
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
```

You can also reduce the generated image resolution.

For example, instead of:

```python
width = 512
height = 768
```

try:

```python
width = 448
height = 640
```

---

## Incomplete Model Download

If the error says that files such as:

```text
unet/diffusion_pytorch_model.safetensors
```

are missing, the local model snapshot is incomplete.

Run:

```bash
python download_wd15.py
```

again to complete the download.

---

# 16. Git and Model Files

Large model files should not be committed to Git.

The `.gitignore` includes:

```gitignore
models/
*.ckpt
*.safetensors
*.bin
*.pt
*.pth

outputs/
```

This keeps the repository lightweight.

Users should download model weights separately.

---

# 17. Development Roadmap

Current progress:

- [x] Stable Diffusion 1.5 inference
- [x] Stable Diffusion pipeline inspection
- [x] Random seed experiments
- [x] Prompt engineering
- [x] Negative prompt experiments
- [x] Anime portrait generation
- [x] Full-body anime generation
- [x] Anime-specific diffusion checkpoint
- [x] Memory-efficient inference
- [ ] Sampling-step comparison
- [ ] Guidance-scale experiment
- [ ] LoRA fine-tuning
- [ ] Small custom image dataset
- [ ] ControlNet
- [ ] Pose-controlled generation
- [ ] Image-to-image generation
- [ ] IP-Adapter
- [ ] Image-to-video generation
- [ ] Video Diffusion
- [ ] Temporal consistency evaluation
- [ ] Audio-conditioned video generation
- [ ] Music-driven controllable video generation

---

# 18. Future Direction

The next development stages will move from image generation toward video generation.

Planned technical path:

```text
Stable Diffusion
       ↓
LoRA Fine-tuning
       ↓
ControlNet
       ↓
Image-to-Image
       ↓
Image-to-Video
       ↓
Video Diffusion
       ↓
Temporal Modeling
       ↓
Temporal Consistency
       ↓
Audio / Music Conditioning
       ↓
Controllable Video Generation
```

The long-term objective is to study lightweight multimodal generative AI systems for image and video creation on consumer hardware.

---

# 19. Disclaimer

This repository is intended for:

- Educational use
- Research experiments
- Generative AI learning
- Algorithm prototyping

Third-party model weights are not redistributed in this repository.

Users should obtain model weights from their original sources and comply with the corresponding licenses and terms of use.
