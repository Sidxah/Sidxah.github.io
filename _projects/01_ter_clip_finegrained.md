---
layout: page
title: Fine-Grained Text Supervision in CLIP-like Models
description: TER research project at CEA LIST, exploring how specialized captions reshape vision-language alignment.
img:
importance: 1
category: "Vision & Multimodal AI"
related_publications: true
---

## Summary

This is my **current research project (TER)**, conducted at the [CEA LIST](https://list.cea.fr/) LASTI Laboratory under the supervision of [Prof. Adrian Popescu](https://adrianpopescu.github.io/), in collaboration with PhD candidate Mehdi Zakaria Adjal.

A manuscript is in preparation; this page summarises the motivation and approach. Code and full results will follow once the report is finalised.

## Motivation

Vision-language models like CLIP ([Radford et al., 2021](https://arxiv.org/abs/2103.00020)) learn powerful representations by aligning images with their captions through a contrastive objective. The captions used at training time are typically short, web-scraped, and generic, of the form *"a dog on a beach"* or *"a black car"*. This is enough to recover broad semantic categories. What happens when the supervision is far more **fine-grained**, when each caption distinguishes a *Border Collie* from a *Bernese Mountain Dog*, or *late-19th-century Impressionism* from *post-Impressionism*, is much less well understood.

A growing body of work suggests that fine-grained supervision matters for downstream tasks ([SEAL](https://arxiv.org/abs/2402.03293), [BioCLIP](https://arxiv.org/abs/2311.18803)), but the question of *how much* and *under what regime* it matters in CLIP-style architectures remains under-explored, partly because such annotations are scarce.

## What we are studying

We investigate three complementary questions:

1. **Representation quality.** Does fine-grained supervision change the *geometry* of the learned space, or only the alignment scores at the top of training?
2. **Few-shot transfer.** Does it improve performance on downstream tasks where the supervision signal is sparse, i.e. where the model needs to do most of the work itself?
3. **Robustness.** Does it make representations more sensitive to specialised vocabulary at inference, or does it generalise to natural language more broadly?

## Methodology

The pipeline trains CLIP-style dual-encoders on standard datasets and on pairs that have been re-captioned with progressively richer descriptions. We then evaluate on:

- Zero-shot classification (CIFAR-10/100, ImageNet-V2, fine-grained datasets).
- Linear-probe accuracy as a proxy for representation quality.
- Few-shot transfer to specialised domains.

Comparisons are made against the original CLIP baseline and against models with controlled-noise captions to isolate the *fine-grained* signal from the *length* signal.

## Status

- Literature review completed (CLIP, [SimCLR](https://arxiv.org/abs/2002.05709), [DINO](https://arxiv.org/abs/2104.14294), [MAE](https://arxiv.org/abs/2111.06377), [I-JEPA](https://arxiv.org/abs/2301.08243), [V-JEPA](https://arxiv.org/abs/2404.08471), SEAL, BioCLIP).
- Re-implementation of CLIP from scratch (see [related project](/projects/01_clip_from_scratch/)) as a clean baseline.
- Training infrastructure and evaluation harness in place.
- Final experiments and writing in progress.

## Acknowledgments

Carried out at CEA LIST, LASTI Lab, Université Paris-Saclay. Many thanks to Prof. Adrian Popescu and Mehdi Zakaria Adjal for their guidance throughout this project.
