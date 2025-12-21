---
layout: page
title: CLIP from Scratch
description: Reproducing OpenAI's CLIP model to understand multimodal representation learning
img: 
importance: 1
category: Research
github: https://github.com/Sidxah
---

## Overview

This project aims to reproduce the **CLIP (Contrastive Language-Image Pre-training)** model from [Radford et al. (2021)](https://arxiv.org/abs/2103.00020) to deeply understand vision-language alignment through contrastive learning.

## Motivation

CLIP represents a paradigm shift in computer vision by learning visual representations from natural language supervision. Understanding its internals is crucial for my upcoming TER on fine-grained textual descriptions in multimodal learning.

## Architecture

The model consists of:
- **Image Encoder**: Modified ResNet-50 with attention pooling
- **Text Encoder**: Transformer (GPT-2 style) with 6 layers
- **Contrastive Loss**: InfoNCE loss for aligning image-text pairs

```python
# InfoNCE Loss
L = -log(exp(sim(I_i, T_i) / τ) / Σ_j exp(sim(I_i, T_j) / τ))
```

## Current Progress

- [x] Image Encoder (ResNet-50)
- [x] Text Encoder (Transformer)
- [x] InfoNCE Loss implementation
- [ ] Training on Flickr30k
- [ ] Zero-shot evaluation on CIFAR-10

## Technologies

`PyTorch` `torchvision` `Hugging Face Transformers` `Contrastive Learning`

## Status

🔄 **In Progress** — Expected completion: January 2026
