---
layout: page
title: CLIP from Scratch
description: A complete PyTorch implementation of CLIP with educational documentation
img: assets/img/clip_architecture.png
importance: 1
category: research
github: https://github.com/Sidxah/CLIP-from-scratch
---

## Overview

**CLIP** (Contrastive Language-Image Pre-training) learns to connect images and text through contrastive learning. This project is a complete from-scratch implementation designed as both a learning resource and research baseline.

## Motivation

I built this project to deeply understand vision-language alignment before starting my TER research on fine-grained text supervision in CLIP-like models at CEA LIST.

## Architecture

```
Image (224×224×3)                    Text (token IDs)
       │                                    │
       ▼                                    ▼
┌──────────────┐                    ┌──────────────┐
│   ResNet-18  │                    │ Transformer  │
│  (11.2M params)│                  │ (38.5M params)│
└──────────────┘                    └──────────────┘
       │                                    │
       ▼                                    ▼
   [512-dim]                            [512-dim]
       │                                    │
       └────────────┬───────────────────────┘
                    │
                    ▼
            Cosine Similarity
                    │
                    ▼
           Contrastive Loss
```

## Key Components

### Image Encoder (ResNet)
- Implemented from scratch with residual blocks
- Skip connections enable training deep networks
- Global average pooling → projection to embedding space

### Text Encoder (Transformer)
- Multi-head self-attention mechanism
- Causal masking (autoregressive)
- Last token embedding represents the sequence

### Contrastive Learning
- Symmetric InfoNCE loss
- Learnable temperature parameter
- Batch size = number of negative examples

## Results

| Dataset | Zero-Shot Accuracy |
|---------|-------------------|
| CIFAR-10 | **76.2%** |
| CIFAR-100 | 42.1% |

## What I Learned

1. **Contrastive learning is elegant**: The idea of learning by comparison is simple yet powerful
2. **Temperature matters**: Lower temperature → sharper predictions, but harder to train
3. **Batch size is crucial**: More negative examples = better representation learning
4. **Vision-language alignment is non-trivial**: Getting the two modalities to "speak the same language" requires careful architecture design

## Code

The entire implementation is available on [GitHub](https://github.com/Sidxah/CLIP-from-scratch) with detailed documentation explaining every component.

```python
# Example: Zero-shot classification
from src.model import CLIP
from src.evaluate import ZeroShotClassifier

model = CLIP()
classifier = ZeroShotClassifier(model, tokenizer, ["cat", "dog", "car"])
predictions = classifier.predict(images)
```

## References

- Radford et al. "Learning Transferable Visual Models From Natural Language Supervision" (2021)
- He et al. "Deep Residual Learning for Image Recognition" (2015)
- Vaswani et al. "Attention Is All You Need" (2017)
