---
layout: page
title: "PyxiScience: LLM Evaluation & Production Content Pipeline"
description: Building, benchmarking and deploying the LLM-driven math-exercise generation pipeline that powers PyxiScience, an adaptive learning platform now used by 50k+ students at institutions including Sorbonne Paris Nord and NYU Paris.
img:
importance: 1
category: "AI Engineering & Tools"
---

## Context

Since August 2025 I work as **AI Research Engineer (CDD)** at [PyxiScience](https://pyxiscience.com), an AI-powered adaptive learning platform incubated at **Station F (Paris)** and **StartX (Stanford)**. My supervisor is **Jacques Lévy Véhel**, former INRIA Research Director (4,200+ citations), who co-founded the company with **Joachim Lebovits**. The platform is now used by **50k+ students** at institutions including **Sorbonne Paris Nord** and **NYU Paris**, with mathematics exercises generated and validated by the LLM-driven content pipeline I work on.

This is the side of my profile that is closest to applied AI engineering, a useful counterweight to my research projects, and a place where I have learned a great deal about *what actually breaks* between a clean prototype and a production system.

## What I built

### LLM evaluation framework
A systematic comparison of **GPT-4**, **Gemini** and **DeepSeek** on mathematical reasoning and exercise-generation tasks, scored on three axes that I had to define from scratch for educational content:

- **Accuracy.** Does the generated solution actually solve the problem?
- **Hallucination rate.** How often does the model invent steps that look right but are wrong?
- **Inference cost.** Measured per validated exercise, not per token.

The result is an internal scoreboard that drives provider choice per task, and a methodology I can reuse on new models without rewriting the harness.

### Two-stage prompt engineering methodology
A process I built to separate concerns in prompt design:

- **Stage 1.** Pedagogical constraints: difficulty calibration, curriculum alignment, the kind of mistake the exercise is meant to catch.
- **Stage 2.** Technical formatting: LaTeX, MyST, validator-friendly structure.

Combining them in a single mega-prompt produced inconsistent quality. Splitting them and chaining the two stages with intermediate validation improved content quality by **~40 %** measured against the internal rubric.

### AST-based hallucination detection
For mathematical content, *plausible-looking* output is the dangerous failure mode. I built a static analyser that parses generated Python code (used in exercise solvers) into an AST and checks structural invariants: variable definitions, type consistency, presence of mandatory return values, edge-case handling. This automated layer caught ~60 % of cases that previously required manual review.

### Parallel async generation pipeline
A Python pipeline using `asyncio` to fan out hundreds of concurrent LLM requests across providers, with rate-limit-aware backoff, partial-failure recovery, and structured logging for debugging. The throughput improvement versus sequential generation is roughly the number of providers we run in parallel (three, in practice).

## What this enables for teachers

The exercise-generation pipeline is the upstream layer that makes the teacher-facing side of the platform tractable. Where a teacher would otherwise hand-author dozens of variants of the same exercise to cover difficulty levels, common mistakes and curriculum constraints, the pipeline produces those variants automatically from a single seed exercise, with pedagogical metadata attached. Combined with the platform's AI-assisted grading layer (90% accuracy on handwritten papers and up to 70% of grading time saved per [PyxiScience's published figures](https://pyxiscience.com)), the result is a workflow that lets a teacher prepare and grade differentiated material at the scale of a real cohort, not at the scale of what one human can manually write. My work sits squarely on the generation side of that loop.

## Why this work matters to me beyond engineering

Two reasons.

First, working on **LLM evaluation** day-to-day teaches you that *the metric is the product*. The same LLM looks brilliant or useless depending on what you score it on. This habit of being suspicious of headline benchmarks transfers directly to research: the [EPFL Hackathon](/projects/10_epfl_quantum_hackathon/) lesson, that LightGBM was first on ΔPC space and last on real volatility, is the same lesson I learn over and over here, on a different scale.

Second, working in **production at scale** gives me a sense of the engineering side of large-model systems that pure research training does not provide. When models go to FAIR Paris or DeepMind, they need to *ship*. That mindset is not antithetical to research, it is complementary.

## Tech stack

`Python` · `asyncio` · `OpenAI / Gemini / DeepSeek APIs` · `LangChain` · `RAG (FAISS, HuggingFace embeddings)` · `Prompt engineering` · `LaTeX / MyST validation` · `AST-based static analysis`

## Acknowledgments

Thanks to **Jacques Lévy Véhel** (former INRIA RD) and **Joachim Lebovits** for taking me on, and for letting me run experiments on real production traffic.
