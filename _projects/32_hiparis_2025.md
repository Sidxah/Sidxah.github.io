---
layout: page
title: "Hi! Paris Hackathon 2025: Top 3 of 400+"
description: PISA Education Innovation Challenge at the Hi! Paris Hackathon 2025 (École Polytechnique × HEC). Two-deliverable submission combining a stacked ML pipeline (R² ≈ 0.77 on the leaderboard) with an AI-powered education product proposal targeting the 23.4% high-potential / low-motivation segment.
img:
importance: 3
category: "AI Engineering & Tools"
github: https://github.com/Sidxah/HiParis-Hackathon-2025
---

## Top 3 of 400+ teams

[Repository on GitHub](https://github.com/Sidxah/HiParis-Hackathon-2025) (ML pipeline + EBR pitch).

The 2025 edition of the Hi! Paris Hackathon, jointly organised by **École Polytechnique** and **HEC Paris**. Our team finished **3rd of more than 400 teams** on the PISA Education Innovation Challenge.

## The challenge

Given a 1.7M-row anonymised slice of the **PISA international assessment** (15-year-olds, 80+ countries, hundreds of features per student), the Innovation Challenge asked two things at once: (i) build a predictive model of student MathScore, and (ii) translate the EDA into a credible product or policy proposal targeting a real educational gap.

The catch on the modelling side: more than **50% of cells are missing**, and the missingness is structured rather than random, *the* PISA modelling problem in miniature. The catch on the product side: avoid the obvious "AI tutor" answer and find a target segment whose constraints are not purely academic.

## What we built

The submission combined two deliverables that shared the same PISA EDA spine.

### Deliverable 1: Stacked ML pipeline (R² ≈ 0.77)

A reproducible pipeline targeting MathScore prediction at the leaderboard level.

- **Differentiated imputation.** Per-feature strategies driven by the psychometric nature of each variable: median imputation for skewed psychological items (ST208, ST273, ST268), mean imputation for wide scales (ST290, ST301), median for behavioural variables (IC177 screen time). Preserved more than 95% of the dataset while respecting the original psychometric distributions.
- **Ensemble core.** LightGBM, XGBoost and CatBoost trained independently with **Optuna** Bayesian search, then combined via a **stacking ensemble** with a Ridge meta-learner trained on out-of-fold predictions to avoid leakage.
- **Distribution-shift control.** Adversarial validation between training and official test set; AUC-driven filtering of training samples that did not look like the test distribution.
- **Cluster-conditioned modelling.** K-Means on a standardised, imputed copy of the features, then a CatBoost per cluster, useful where the response surface differs across student profiles.
- **Neural error correction.** A small MLP trained on residuals (positive/negative wellbeing indicators standardised, reverse-coded as appropriate) to correct systematic mis-predictions of the boosting layer.
- **Interpretability.** SHAP analysis on the final CatBoost to surface the features that actually drove individual predictions, not just global importance.

Result on the public leaderboard: **R² ≈ 0.77**, supporting the **Top 3 / 400+** ranking.

### Deliverable 2: Educational Brain Rot (innovation pitch)

The same multivariate PISA EDA also surfaced a target segment we centred the product proposal on: **23.4% of students with high academic potential (MathScore above the median) but low motivation (ST208 below the median)**, who report on average more classroom disruptions and 5.2 hours of daily digital media use against a cohort average of 4.1 hours.

The proposal, *Educational Brain Rot* (EBR), is a personalised micro-content layer that delivers 15–60-second mathematics and computer-science clips in viral formats native to the platforms this segment already uses (TikTok, Instagram, YouTube Shorts). The personalisation pipeline takes the student's psychometric profile (motivation ST208, confidence ST290, classroom-disruption signal ST273) and an n8n-orchestrated workflow calling Mistral generates a tailored script, a difficulty target, a quiz frequency and a tone register before the clip is rendered. The thesis: the deficit in this segment is not in attention capacity, it is in attention direction, and existing recommendation algorithms can be turned into the distribution channel rather than fought as the enemy.

The full pitch is documented in the team's submitted brief (data-driven targeting, imputation strategy, AI personalisation pipeline, B2B/B2C operating model, EU SAM sizing of 9.4M target students).

## Why both deliverables together mattered

The PISA dataset is large and noisy enough that almost any team could produce a model with a respectable R². The differentiator was using the same EDA to identify a credible target segment with measurable constraints (not "students who struggle in maths" but "students who already sustain hours of attention online and have above-median academic potential"), then designing a delivery format that respects the target's actual behaviour rather than fighting it. The jury read the submission as a coherent pipeline from data to product, not as two unrelated pieces.

Lesson that compounded across this hackathon: **the gap between a top-50 team and a top-3 team is not architectural sophistication, it is the discipline of validation** (stratified k-fold by country and socio-economic decile, adversarial validation, faithful CV) and the willingness to commit to a non-obvious target segment in the product brief.

## Tech stack

`Python` · `pandas` · `scikit-learn` · `LightGBM` · `XGBoost` · `CatBoost` · `Optuna` · `SHAP` · `PyTorch` (residual MLP) · `n8n` (personalisation workflow) · `Mistral API`

## Team

Sid Ahmed Bouamama (Université Paris-Saclay), Chahyn Ettaghi (INP-ENSEEIHT), Adrien Etienne (Télécom SudParis), Wilen Yaici (Université Paris-Saclay), Yanis Takbou (ENSTA Paris), Gordan Kongue (ENSTA Paris).
