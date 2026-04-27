---
layout: page
title: CLIP from Scratch
description: A complete, well-documented PyTorch reimplementation of OpenAI's CLIP, built as a baseline for my TER research and as a study of contrastive vision-language alignment.
img:
importance: 2
category: "Vision & Multimodal AI"
github: https://github.com/Sidxah/CLIP-from-scratch
---

## Overview

A from-scratch implementation of **CLIP** (*Contrastive Language–Image Pre-training*, [Radford et al., 2021](https://arxiv.org/abs/2103.00020)), written in PyTorch with educational documentation explaining every component. Built in preparation for my [TER on fine-grained text supervision](/projects/01_ter_clip_finegrained/) at CEA LIST.

[Code on GitHub →](https://github.com/Sidxah/CLIP-from-scratch)

## Why rebuild CLIP

CLIP marked a paradigm shift in computer vision: instead of training on a fixed label set, it learns visual representations from natural-language supervision. Re-implementing it from scratch, rather than using `transformers.CLIPModel`, was the most reliable way for me to understand what each architectural choice does, what each hyperparameter changes, and where the actual difficulty hides (the answer is *batch size and temperature*).

## Architecture

<figure style="margin: 1.5em 0; text-align: center;">
  <img src="{{ '/assets/img/projects/clip-architecture.svg' | relative_url }}" alt="CLIP architecture: a ResNet-18 image encoder and a Transformer text encoder, each projecting to a 512-dimensional embedding, joined by a cosine similarity and a symmetric InfoNCE loss." style="max-width: 100%; height: auto;">
</figure>

## What is in the repo

- A **vision encoder** (modified ResNet) implemented from scratch with residual blocks, attention pooling and projection head.
- A **text encoder** (causal Transformer) built layer by layer, including byte-pair tokenisation.
- The **symmetric InfoNCE loss** with a learnable temperature parameter.
- A **training loop** on Flickr30k with mixed-precision and gradient accumulation.
- A **zero-shot evaluation harness** running on CIFAR-10 / CIFAR-100.

## Results

| Dataset    | Zero-shot accuracy |
|------------|--------------------|
| CIFAR-10   | **76.2 %**         |
| CIFAR-100  | 42.1 %             |

These numbers are not state-of-the-art (a small encoder + small dataset will not match OpenAI's WIT-400M run), but they confirm the implementation works end-to-end and reproduces the qualitative behaviour described in the original paper.

## Three things I learned

1. **Batch size is the loss.** With InfoNCE, the number of negatives per positive scales with the batch. On a single GPU, this is the bottleneck, and gradient accumulation does not fully fix it.
2. **Temperature is fragile.** Too low and the softmax becomes one-hot before the model has learned anything; too high and the gradient signal vanishes. Making it learnable (CLIP's choice) is the right move.
3. **Vision–language alignment is not symmetric.** Image-to-text retrieval works well long before text-to-image retrieval becomes competitive. The representations of the two modalities settle into the shared space at different speeds.

## References

- Radford et al., *Learning Transferable Visual Models from Natural Language Supervision*, 2021.
- He et al., *Deep Residual Learning for Image Recognition*, 2015.
- Vaswani et al., *Attention Is All You Need*, 2017.
- van den Oord et al., *Representation Learning with Contrastive Predictive Coding* (InfoNCE), 2018.
