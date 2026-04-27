---
layout: page
title: "Sampling-Quality Trade-offs in Denoising Diffusion Probabilistic Models"
description: A from-scratch reproduction study of DDPMs on CIFAR-10 under a fixed 100-epoch training budget, with a T-ablation, a deterministic DDIM evaluation, and a per-timestep noise-prediction MSE diagnostic.
img:
importance: 4
category: "Vision & Multimodal AI"
---

## Summary

A solo, from-scratch reproduction of the [DDPM formulation of Ho et al. (2020)](https://arxiv.org/abs/2006.11239) on CIFAR-10, under a fixed 100-epoch training budget on a single V100. The study asks two questions that practitioners face but the original paper does not ablate: how does the number of diffusion timesteps T arbitrate the quality-speed trade-off in this regime, and to what extent can [DDIM (Song et al., 2021)](https://arxiv.org/abs/2010.02502) decouple sampling cost from quality at inference?

Final project of the **Deep Learning** course of the M1 in Quantum & Distributed Computer Science at Université Paris-Saclay (Prof. Thomas Gerald, LISN / CNRS), submitted as an ICLR-style technical report. Code and report under final review before release. Conducted March to April 2026.

## Setup

A compact 8.93M-parameter U-Net (base width 64, multipliers `[1, 2, 2, 2]`, two residual blocks per level, single self-attention layer at 16×16, GroupNorm + SiLU + dropout 0.1, sinusoidal timestep embeddings of dimension 256). Linear β-schedule with β_min = 10⁻⁴ and β_max = 2·10⁻². Adam, learning rate 2·10⁻⁴, batch size 128, no learning-rate schedule and no warmup, EMA decay 0.9999 used at sampling time. The number of optimiser steps per epoch is fixed at ⌈50 000 / 128⌉ = 390 regardless of T, so the three runs see the same number of gradient updates.

FID is computed with the reference `pytorch-fid` implementation against 10 000 generated samples and 10 000 real CIFAR-10 training images, using Inception-V3 pool3 features.

## Findings

**The DDPM ablation across T saturates around T=500.** Doubling the schedule from T=500 to T=1000 leaves the FID essentially unchanged while doubling the sampling cost. Halving to T=200 trades 26 FID points for a roughly 5× sampling speed-up, which is attractive only in latency-bound applications.

| T    | Final loss | FID ↓ | Sampling time (64 images) |
|------|:----------:|:-----:|:-------------------------:|
| 200  | 0.0662     | 64.52 | 3.68 s                    |
| 500  | 0.0427     | **37.64** | 9.19 s                |
| 1000 | 0.0305     | 39.23 | 18.34 s                   |

**The deterministic DDIM sampler does not rescue inference at this training budget.** Re-using the trained T=1000 model with η=0 at five sampling budgets S ∈ {10, 25, 50, 100, 250}, the FID curve plateaus around 90 for S ≥ 50, more than 50 FID points above the DDPM(T=1000) baseline at 39. The standard "DDIM as a free speed-up" narrative implicitly assumes a well-trained noise predictor; in the under-trained regime, removing the stochastic correction √β̃ₜ z appears to expose accumulated per-step prediction error that DDPM was implicitly compensating for.

| S         | FID ↓  | Sampling time (64 images) | Speed-up vs DDPM(T=1000) |
|-----------|:------:|:-------------------------:|:------------------------:|
| 10        | 117.08 | 0.19 s                    | 96×                      |
| 25        | 96.01  | 0.54 s                    | 34×                      |
| 50        | 91.54  | 0.97 s                    | 19×                      |
| 100       | 90.27  | 2.12 s                    | 9×                       |
| 250       | 90.70  | 4.95 s                    | 4×                       |
| DDPM, T=1000 | **39.23** | 18.34 s              | (baseline)               |

**The average per-step training loss is intrinsically not comparable across different T.** A per-timestep diagnostic on a held-out batch of 1024 CIFAR-10 test images shows that the noise-prediction MSE varies by **more than three orders of magnitude across t**, from 0.62 at t=0 to 5·10⁻⁴ at t=990. The uniform-t mean (0.034 in the T=1000 model, close to the average reported during training) is dominated by the easy regime at large t where the input is essentially pure noise and almost any prediction has small ℓ₂ error. Reading lower training MSE as "better trained" across different T is therefore an artefact of the integration grid.

## Why this matters

Three takeaways follow from the numbers above. First, the standard recommendation T=1000 is wasteful at this training budget: the operating point with the best quality-per-step is T=500, and the choice should be reconsidered every time one moves the training budget. Second, the speed-up offered by DDIM is not free in this regime: for any FID target reachable by either family, DDPM at T=500 (37.6 FID at 9.2 s per 64-image batch) dominates every DDIM operating point measured. The classical claim that DDIM yields roughly equal quality at much lower cost should therefore be qualified by a training-budget condition. Third, the per-timestep MSE profile motivates a non-uniform t-sampling distribution biased toward small t, where most of the network's actual prediction error sits and where additional gradient updates would have the largest marginal effect on FID.

## Limitations and follow-ups

The training budget (100 epochs, single seed) is one order of magnitude smaller than the original DDPM paper, which is why all absolute FID values are an order of magnitude higher than published state-of-the-art on CIFAR-10. The 1.6-FID gap between T=500 and T=1000 should be interpreted as within-noise on a single seed; the DDIM negative result (gap > 50 FID) is robust to the noise floor. The natural next checks are: (i) the cosine schedule and learnable variances of [Nichol & Dhariwal (2021)](https://arxiv.org/abs/2102.09672), known to make DDIM more robust at small S; (ii) a controlled study of DDIM with η ∈ [0, 1] to localise how much stochasticity is needed to recover the DDPM FID; (iii) a non-uniform t-sampling distribution as suggested by the per-timestep diagnostic.

## Acknowledgments

Solo project conducted for the M1 QDCS Deep Learning course at Université Paris-Saclay, supervised by **Prof. Thomas Gerald** (LISN, CNRS). Compute on a single NVIDIA Tesla V100 32 GB.
