# Lightweight Generative AI

A lightweight hands-on project for exploring diffusion-based generative AI on consumer GPUs.

The project currently focuses on Stable Diffusion inference, prompt engineering, anime image generation, and memory-efficient inference, with future extensions toward LoRA, ControlNet, and video generation.

## Models

Two diffusion checkpoints are currently used:

```text
stable-diffusion-v1-5/stable-diffusion-v1-5
```

Used in the basic Stable Diffusion experiments, including pipeline inspection, seed comparison, prompt engineering, and anime-style generation.

```text
waifu-diffusion/wd-1-5-beta2
```

Used specifically in the Waifu Diffusion experiment:

```text
07_waifu_diffusion.py
```

## Environment

Tested on:

```text
Windows 11
Python 3.10
PyTorch 2.5.1
CUDA 12.4
NVIDIA RTX 3070 Laptop GPU (8GB)
```

## Installation

Clone the repository:

```bash
git clone https://github.com/jackrh2019/Lightweight-Generative-AI.git
cd Lightweight-Generative-AI
```

Create the environment:

```bash
conda create -n aigc_sd python=3.10 -y
conda activate aigc_sd
```

Install PyTorch:

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

Install other dependencies:

```bash
pip install -r requirements.txt
```

## Download Waifu Diffusion

The Stable Diffusion 1.5 experiments can download the model through Hugging Face when first executed.

For the Waifu Diffusion experiment, run:

```bash
python download_wd15.py
```

The checkpoint will be stored in:

```text
models/wd-1-5-beta2/
```

Large model weights are not included in this Git repository.

## Experiments

### Stable Diffusion 1.5

Inspect the Stable Diffusion pipeline:

```bash
python 02_inspect_pipeline.py
```

Run random seed experiments:

```bash
python 03_seed_experiment.py
```

Run prompt / image generation experiments:

```bash
python 04_prompt_experiment.py
```

Generate anime portraits:

```bash
python 05_anime_portrait.py
```

Generate full-body anime images:

```bash
python 06_anime_fullbody.py
```

These experiments use:

```text
stable-diffusion-v1-5/stable-diffusion-v1-5
```

### Waifu Diffusion

Run:

```bash
python 07_waifu_diffusion.py
```

This experiment uses:

```text
waifu-diffusion/wd-1-5-beta2
```

Generated images are saved under:

```text
outputs/
```

## Memory Optimization

For GPUs with limited VRAM:

```python
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
```

## Roadmap

- [x] Stable Diffusion 1.5 inference
- [x] Pipeline inspection
- [x] Seed experiments
- [x] Prompt engineering
- [x] Anime-style generation with SD 1.5
- [x] Waifu Diffusion checkpoint experiment
- [ ] LoRA fine-tuning
- [ ] ControlNet
- [ ] Image-to-video generation
- [ ] Video diffusion
- [ ] Audio-conditioned video generation

## Project Direction

```text
Stable Diffusion
      ↓
LoRA
      ↓
ControlNet
      ↓
Image-to-Video
      ↓
Video Diffusion
      ↓
Audio / Music Conditioned Video Generation
```

This repository is intended for educational and research experiments.
