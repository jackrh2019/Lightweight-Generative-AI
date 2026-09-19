# Lightweight Generative AI Lab

A lightweight hands-on project for exploring diffusion-based generative AI on consumer GPUs.

Current work focuses on Stable Diffusion, prompt engineering, anime image generation, and memory-efficient inference, with future extensions toward LoRA, ControlNet, and video generation.

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

## Download Model

Model weights are not included in this repository.

Download the anime diffusion checkpoint:

```bash
python download_wd15.py
```

The model will be stored in:

```text
models/wd-1-5-beta2/
```

## Experiments

Inspect the Stable Diffusion pipeline:

```bash
python 02_inspect_pipeline.py
```

Run random seed experiments:

```bash
python 03_seed_experiment.py
```

Generate anime portraits:

```bash
python 05_anime_portrait.py
```

Run full-body anime generation:

```bash
python 06_anime_fullbody.py
```

Run the Waifu Diffusion experiment:

```bash
python 07_waifu_diffusion.py
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
- [x] Anime image generation
- [ ] Anime-specific checkpoint integration
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
