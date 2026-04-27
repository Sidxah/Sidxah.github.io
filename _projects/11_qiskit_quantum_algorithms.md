---
layout: page
title: Quantum Algorithm Exploration with Qiskit & Perceval
description: Hands-on study of the canonical quantum algorithms (Deutsch–Jozsa, Grover, QFT, Shor, VQE, QAOA) and of quantum error correction codes, in both gate-based and photonic frameworks.
img:
importance: 2
category: "Quantum Computing"
---

## Summary

A self-directed portfolio of canonical quantum algorithms and error-correction codes, implemented as part of the *Quantum Information* and *Quantum Algorithms* courses of my M1 in Quantum & Distributed Computer Science at Université Paris-Saclay, and extended through volunteer involvement in the [Qiskit](https://qiskit.org/) open-source community.

Most implementations are in **Qiskit** for gate-based quantum computing, with a parallel track in **[Perceval](https://perceval.quandela.net/)** (Quandela) for photonic quantum computing.

> *Repository currently being assembled. Cleaned-up notebooks for each algorithm will be released one at a time, and the link will appear here when the first batch is ready.*

## What is in there

### Foundational algorithms
- **Deutsch–Jozsa.** The textbook example of quantum advantage on a black-box query problem.
- **Grover's search.** A step-by-step study of its `O(√N)` scaling and amplitude-amplification machinery.
- **Quantum Fourier Transform (QFT).** The building block behind Shor and phase estimation.
- **Shor's factoring algorithm.** Implemented and tested on small composite numbers.

### Variational and hybrid algorithms
- **VQE (Variational Quantum Eigensolver).** Finding ground-state energies for small Hamiltonians (H₂, LiH); a study of how the parameterised ansatz and the classical optimiser interact.
- **QAOA (Quantum Approximate Optimization Algorithm).** Applied to small Max-Cut and graph-partition instances, with comparisons to classical heuristics.

### Quantum error correction
- The **bit-flip** and **phase-flip** repetition codes, the simplest examples of redundancy on quantum information.
- The **Shor [9, 1, 3] code**, combining bit-flip and phase-flip protection.
- A reading-and-implementation walk-through of **surface codes** ([Fowler et al., 2012](https://arxiv.org/abs/1208.0928)), with a small toy lattice.

### Photonic quantum computing
- Single-photon sources, beam splitters, phase shifters and measurement in **Perceval**.
- Re-implementing Deutsch–Jozsa and small variational circuits in the photonic paradigm.
- The **Quantum Reservoir Computing** experiments developed during the [EPFL Quantum Hackathon](/projects/10_epfl_quantum_hackathon/).

## Why I am doing this

The honest reason: I want to be in a position to evaluate, and eventually contribute to, **Quantum Machine Learning** research without taking the field's claims at face value. There is real signal in QML (the [Quandela Quantum Optical RC](https://arxiv.org/abs/2512.08318) preprint and [Li et al. (2025)](https://arxiv.org/abs/2505.13933) suggest non-trivial advantages on specific workloads), and there is also a lot of noise. Building the algorithms by hand is the surest way to tell the two apart.

## References

- Nielsen & Chuang, *Quantum Computation and Quantum Information*, Cambridge University Press.
- Preskill, *Quantum Information* lecture notes, Caltech ([online](http://theory.caltech.edu/~preskill/ph229/)).
- Fowler, Mariantoni, Martinis & Cleland, *"Surface codes: Towards practical large-scale quantum computation"*, [arXiv:1208.0928](https://arxiv.org/abs/1208.0928).
- IBM Qiskit textbook, [qiskit.org/learn](https://qiskit.org/learn/).
