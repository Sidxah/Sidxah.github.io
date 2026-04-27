---
layout: page
title: "Fake News Detection on LIAR2: A Multimodal NLP Study"
description: A progressive study from TF-IDF and Word2Vec to DeBERTa-v3 fine-tuning and Late Fusion, showing why metadata beats transformers when the text is too short to carry the signal.
img:
importance: 3
category: "Vision & Multimodal AI"
github: https://github.com/Sidxah/M1-NLP-LIAR2-FakeNews-Detection
---

## Summary

A semester-long M1 NLP project at Université Paris-Saclay (Hands-on NLP course, Profs. Kim Gerdes & Nona Naderi), conducted with Aya Messaoudi. We frame fake news detection on the [LIAR2 dataset](https://huggingface.co/datasets/chengxuphd/liar2) as a **multimodal problem**. The question is not *"can a transformer detect lies in 18 words?"* (the answer is largely no) but *"what does the model actually need beyond the text?"*.

[Full code, README and notebook on GitHub →](https://github.com/Sidxah/M1-NLP-LIAR2-FakeNews-Detection)

## The key finding

> **A 2013 Word2Vec + Random Forest model beats a 2021 DeBERTa-v3 fine-tune by +7 F1 points** on this task.

Not because Word2Vec is better than DeBERTa, but because the actual signal in LIAR2 is in the **speaker's credit history** (six counts of how many times they have lied before), and a transformer fed only an 18-word political statement has no way to access that. The transformer's tokenizer even splits the integer `243` into sub-word tokens, destroying its numerical meaning. We fix this with a **Late Fusion architecture** that gives the transformer the text and an MLP the structured metadata, then concatenates the two streams before the head.

## Methodology

We progressed from the simplest baselines to research-grade techniques, comparing each stage rigorously:

```
TF-IDF  →  Word2Vec  →  Transformers (RoBERTa, DeBERTa)  →  Late Fusion  →  AutoGluon
~0.54 F1     ~0.49 F1            ~0.59 F1                    ~0.70 F1       ~0.68 F1
```

Techniques layered on top of the basic Late Fusion architecture:

- **LoRA** ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)) for parameter-efficient fine-tuning.
- **Focal Loss** ([Lin et al., 2017](https://arxiv.org/abs/1708.02002)) to address the class imbalance ([0.18, 0.42, 0.40] for False / Mixed / True).
- **Label Smoothing** ([Szegedy et al., 2016](https://arxiv.org/abs/1512.00567)) to reduce overconfidence.
- **Progressive Unfreezing** ([Howard & Ruder, 2018](https://arxiv.org/abs/1801.06146)).
- **AutoGluon Multimodal** as a strong ensemble baseline.

## Selected results

| Model                                    | F1 macro | MCC   |
|------------------------------------------|---------:|------:|
| **DeBERTa-v3 Fusion (Focal + LS)**       | **0.697**| 0.553 |
| AutoGluon Multimodal                     | 0.683    | 0.539 |
| RoBERTa Fusion (Focal + LS)              | 0.681    | 0.534 |
| Logistic Regression on metadata only     | 0.669    | 0.520 |
| Word2Vec + metadata                      | 0.664    | 0.513 |
| TF-IDF + metadata                        | 0.642    | 0.482 |
| **DeBERTa-v3 full fine-tune (text only)**| 0.590    | 0.403 |
| RoBERTa full fine-tune (text only)       | 0.578    | 0.384 |

The take-home: **adding the credit-history features lifts every model by 5 to 10 F1 points**, regardless of which text encoder is used. The architecture matters less than the information you give it.

## Why this matters beyond NLP

This is essentially a controlled study of what happens when a high-capacity model is fed a low-information input, a question that recurs throughout vision-language research. CLIP and its successors face the symmetric problem: when captions are too short or too generic, even a 400M-parameter encoder cannot recover the semantics. It is the same diagnosis I take into my [TER on fine-grained text supervision](/projects/01_ter_clip_finegrained/).

## References

- Wang, *"Liar, Liar Pants on Fire"*, ACL 2017.
- Xu & Kechadi, *"FDHN: Fuzzy Deep Hybrid Network for Fake News Detection"*, IEEE Access 2024.
- He et al., *"DeBERTa: Decoding-enhanced BERT with Disentangled Attention"*, ICLR 2021.
- Hu et al., *"LoRA: Low-Rank Adaptation of Large Language Models"*, ICLR 2022.
