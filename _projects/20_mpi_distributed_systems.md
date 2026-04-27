---
layout: page
title: "MPI & Distributed Algorithms: Portfolio"
description: Implementations and benchmarks of parallel algorithms studied in the M1 HPC track at Paris-Saclay (course by Karim Hasnaoui, IJCLab).
img:
importance: 1
category: "HPC & Distributed AI"
github: https://github.com/Sidxah/M1-HPC-Parallel-Computing
---

## Summary

A portfolio of parallel and distributed algorithms implemented in **MPI**, **OpenMP** and **C++**, drawn from the *MPI Programming* course at Université Paris-Saclay (Prof. Karim Hasnaoui, IJCLab), the *Distributed Algorithms* course (Prof. Sylvie Delaët), and the *Advanced C++* track of the M1 in Quantum & Distributed Computer Science.

Each implementation is paired with a short scaling study so the repo doubles as a personal reference for parallel-systems intuition.

[Repository on GitHub →](https://github.com/Sidxah/M1-HPC-Parallel-Computing)

> *Repository under active assembly. Lab work is being cleaned into a coherent set of mini-projects.*

## Why this matters for AI

Modern foundation models live on parallel hardware. Distributed training, tensor parallelism, FSDP, expert parallelism, async reductions: none of these are sensible without a working mental model of message passing, of strong-vs-weak scaling, of where the communication bottleneck actually lives. **Doing the basics in MPI, by hand, is how that intuition becomes load-bearing rather than performative.**

The same is true on the inference side: efficient serving (DeepSpeed-Inference, vLLM, TGI) is fundamentally a parallel-systems problem, and being comfortable thinking about ranks, communicators and collective operations is non-negotiable.

## What the repository covers

### Parallel computing primitives (MPI)
- **Point-to-point communication.** Blocking and non-blocking sends/receives, deadlock patterns, synchronisation pitfalls.
- **Collective operations.** Broadcast, scatter, gather, all-reduce, reduce-scatter, with measured cost as a function of message size and process count.
- **Topologies.** Cartesian and graph communicators, and their effect on stencil-style computations.

### Numerical algorithms
- **Distributed matrix–matrix multiplication.** Cannon's algorithm and Fox's algorithm on a 2D process grid.
- **Stencil computations** (heat equation, Poisson) with halo exchange.
- **Parallel reduction patterns**, including the difference between tree-based reductions and ring-based all-reduce, which is the algorithmic backbone of distributed gradient averaging in DL.

### Shared-memory parallelism (OpenMP)
- **`#pragma omp parallel for`** with various scheduling strategies (`static`, `dynamic`, `guided`) and their impact on load imbalance.
- **Reductions, atomics, critical sections.** The cost hierarchy.
- **Tasks** for irregular workloads (recursion, graph traversal).
- **Hybrid MPI + OpenMP**, combining process-level and thread-level parallelism on a single node.

### Distributed algorithms (theory side)
- **Self-stabilising algorithms.** Dijkstra's token-ring stabilisation and successors.
- **Leader election, mutual exclusion.** The classic protocols.

### Benchmarking
For each algorithm I record:
- **Strong scaling** (fixed problem, growing process count): efficiency versus ideal.
- **Weak scaling** (problem grows with processes): does throughput hold?
- A short discussion of the dominant cost (computation vs communication) and what that implies for design choices.

## References

- Foster, *Designing and Building Parallel Programs* (the PCAM methodology, still the right mental model).
- Patterson & Hennessy, *Computer Organization and Design*, RISC-V edition.
- Pacheco, *An Introduction to Parallel Programming* (practical MPI).
- The MPI 4.0 standard, [mpi-forum.org](https://www.mpi-forum.org/docs/).
