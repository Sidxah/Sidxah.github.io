---
layout: page
title: "EPFL Quantum Hackathon 2026: 2nd Place"
description: Predicting implied volatility surfaces for swaptions with classical ML and a photonic Quantum Reservoir built in Quandela's Perceval and MerLin.
img:
importance: 1
category: "Quantum Computing"
github: https://github.com/Sidxah/EPFL-Quantum-Hackathon-2026
---

## 2nd Place, Quandela Swaptions Challenge

48 hours at EPFL Lausanne, the first-ever EPFL Quantum Hackathon. We were given **494 days of swaption implied volatility surfaces** (224 values per day, across 14 tenors and 16 maturities) and asked to predict the next 6 days using classical *and* quantum machine learning. Our team finished **2nd**.

[Full repository → README, notebooks, technical defense PDF](https://github.com/Sidxah/EPFL-Quantum-Hackathon-2026)

## The trap of the problem

The data is **so persistent day-to-day** (autocorrelation ≈ 0.98) that a trivial baseline (*"copy yesterday's surface"*) already achieves a MAE of 0.00270, equivalent to 99.87 % accuracy. Any model trying to predict the level of the surface will quickly converge to copying yesterday, because that strategy already minimises the loss. To beat the baseline you have to predict **what changes** between yesterday and today, which is a much weaker signal under a lot of noise.

## What we built: classical pipeline

The guiding principle: **simplify the problem before predicting it.**

<figure style="margin: 1.5em 0; text-align: center;">
  <img src="{{ '/assets/img/projects/epfl-classical-pipeline.svg' | relative_url }}" alt="Classical pipeline for swaption volatility-surface forecasting: raw 224-dim surfaces, PCA to three Level/Slope/Curvature components, first differences to a stationary signal, training-set standardisation, sliding-window framing, parallel evaluation of LSTM, GRU, Ridge, LightGBM and a classical reservoir, then reconstruction back to 224 dimensions." style="max-width: 100%; height: auto;">
</figure>

The three PCA components turn out to be the well-known **Level / Slope / Curvature** modes of the volatility surface, an old result in fixed-income literature ([Litterman & Scheinkman, 1991](https://doi.org/10.3905/jfi.1991.692347)) that emerges naturally from the data.

## Key insight: the metric reverses

The most valuable lesson of this hackathon: **the metric you optimise during training is not the metric that matters.** When ranking by MAE on the normalised ΔPC space, LightGBM wins. When ranking the *same* models by MAE on real volatility (after the full inverse pipeline), the order completely reverses:

| Model                  | Window | Horizon | Ratio (ΔPC) | Ratio (vol.) |
|------------------------|:------:|:-------:|:-----------:|:------------:|
| Naive ("copy yesterday")|   3   |    1    |   1.000     |    1.000     |
| **LSTM**               |   3    |    2    |   0.994     |  **0.997 (best)** |
| GRU                    |   3    |    2    |   0.993     |    1.003     |
| Classical Reservoir    |   5    |    5    |   1.001     |    1.002     |
| LightGBM               |   7    |    3    |   0.967     |    1.030     |
| AutoGluon              |   5    |    2    |   0.988     |    1.058     |

Aggressive models (LightGBM, AutoGluon) make large corrections that look great in the intermediate space but **accumulate errors during reconstruction**. Conservative models (LSTM with `h=2`) make small corrections that survive the inverse PCA. The LSTM is the only classical model that beats the naive baseline in real volatility, by **0.30 %**, small on paper but meaningful in a derivatives book.

## What we built: quantum branch

In parallel we implemented a **photonic Quantum Reservoir Computing** circuit using [Quandela](https://www.quandela.com)'s [Perceval](https://perceval.quandela.net/) and [MerLin](https://merlin.quandela.net/) libraries:

<figure style="margin: 1.5em 0; text-align: center;">
  <img src="{{ '/assets/img/projects/epfl-quantum-reservoir.svg' | relative_url }}" alt="Photonic Quantum Reservoir Computing pipeline: a 3-dimensional input is fed into a 10-mode photonic interferometer parameterised by a Haar-random unitary, three photons are injected, the 120 output Fock probabilities form the reservoir features, and a ridge regression (the only trained component) predicts the next-step delta." style="max-width: 100%; height: auto;">
</figure>

The classical and quantum reservoirs share **the same architecture**, with only the feature extraction differing (`tanh(W·x)` vs photonic interference). The quantum reservoir was running, sampling Fock probabilities, producing features. We did not have time within 48 hours to close the loop between the quantum readout and the full inverse-PCA pipeline.

The open question, whether interference-based features capture correlations that classical random projections miss, is exactly the kind of comparison I want to keep working on. A recent preprint, [Li et al. (2025)](https://arxiv.org/abs/2505.13933), reports that QRC outperforms classical LSTMs for S&P 500 volatility forecasting. Verifying this rigorously on swaptions is unfinished business.

## Tech stack

PyTorch · scikit-learn · LightGBM · AutoGluon · Optuna · Perceval · MerLin

## Acknowledgments

Thanks to **Quandela** for the challenge and access to Perceval and MerLin, **EPFL** for hosting the hackathon, and to my teammates for the late nights.
