# Lightweight Generative AI Lab

A lightweight hands-on project for learning and experimenting with diffusion-based generative AI on consumer GPUs.

The project currently focuses on Stable Diffusion image generation, prompt engineering, sampling experiments, and anime-specific diffusion checkpoints. It is designed to gradually expand toward LoRA fine-tuning, controllable generation, and video diffusion.

The experiments were developed and tested on an NVIDIA RTX 3070 Laptop GPU with 8 GB VRAM.

---

## 1. Project Goals

This repository is built for practical exploration of modern generative AI technologies, including:

- Stable Diffusion
- Latent Diffusion Models
- VAE
- CLIP text conditioning
- U-Net denoising
- Diffusion schedulers
- Prompt engineering
- Negative prompts
- Random seed analysis
- Anime-specific diffusion checkpoints
- LoRA fine-tuning
- Controllable generation
- Image-to-video generation
- Video diffusion

The long-term goal is to build a lightweight generative AI pipeline that can run on consumer GPUs and gradually extend from image generation to controllable video generation.

---

## 2. Current Experiments

### Stable Diffusion 1.5 Inference

Basic text-to-image generation using Stable Diffusion 1.5.

Pipeline:

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
