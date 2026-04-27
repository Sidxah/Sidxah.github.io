---
layout: page
title: "Dragon's Den AI: Multi-Agent Pitch Evaluation"
description: A research-driven AI product for early-stage startup evaluation. Three agent personas with distinct investment theses listen to a founder pitch, debate, and converge on a verdict. Bootstrapped at HackEurope 2026, in active development.
img:
importance: 2
category: "AI Engineering & Tools"
---

## Summary

Dragon's Den AI is a real-time multi-agent system in which three AI venture capitalists with distinct investment theses listen to a founder pitch, challenge the founder publicly, react to one another's reasoning, and converge on a verdict (in / out, equity counter-offer, conviction trajectory). The first prototype was bootstrapped solo over 24 hours at HackEurope 2026 (Paris, Feb 21–22) and is now under active development as a research-driven product.

The code repository is private during active development. Source, demo videos and design notes are available on request (see contact details on the [About page](/)).

## The three agent personas

Each agent is parameterised by a distinct investment thesis encoded in a system prompt of roughly 500 lines:

- **Metrics Maven.** Unit economics, CAC/LTV, gross margins. Benchmarks every claim against a reference set of past pitches.
- **Visionary Contrarian.** Market timing, founder-market fit, ten-year vision. Bets on narratives, not spreadsheets.
- **Growth Hacker.** Viral loops, distribution channels, scalability. Resistant to founder-dependent playbooks.

Critically, the agents *listen to one another*: each receives the full conversation with proper role attribution, can reference what the other dragons said, and can change its position mid-session. Cross-agent reactions emerge organically from the conversation, not from an orchestration script.

## System design

The non-trivial parts of making three LLM agents *behave like a panel rather than a chatbot polyphony*:

- **Phase progression without a script.** Each agent self-reports its conversational phase via inline `[PHASE:]` tags; the session phase is the majority vote across agents. No hard-coded turn counter.
- **Conviction tracking.** Each agent maintains a numeric conviction score in 0–10 that updates continuously; the score is exposed as a streaming channel to the frontend.
- **Verdict triggering.** Agents emit `[READY_FOR_VERDICT: YES]` when they consider their position stable; verdicts fire when two of three are ready, with a hard cap of twelve questions to prevent infinite loops.
- **Negotiation.** Counter-offers (*"€200K for 10 %, not 12 %"*) emerge from the prompt design, not from the orchestration layer.

## Architecture

<figure style="margin: 1.5em 0; text-align: center;">
  <img src="{{ '/assets/img/projects/dragons-den-architecture.svg' | relative_url }}" alt="Dragon's Den AI architecture: a Next.js frontend (pitch form, live chat, verdicts) communicates with a FastAPI backend (orchestrator, 3-agent loop, tag parser, live web search) via an HTTP POST to /api/sessions and a WebSocket return channel that streams tokens, convictions, phase changes and verdicts; the backend in turn calls Claude Sonnet 4.5 with three ~500-line system prompts under prompt caching." style="max-width: 100%; height: auto;">
</figure>

## Tech stack

- **Backend.** FastAPI + WebSockets, `asyncio` orchestration, Claude Sonnet 4.5 with prompt caching (≈ 90 % reduction on repeated system prompts).
- **Frontend.** Next.js 16 (App Router + Turbopack), Framer Motion, Tailwind CSS with a custom `den-*` design system.
- **Live web search.** Agents call out to research the founder's market in real time, grounding their critique in current funding-round data and competitor activity.

## AI research roadmap

The prototype works as a system; the open questions are the ones that turn it into a credible evaluation tool. The directions I am pursuing:

1. **Calibration against published outcomes.** Build a held-out set of past Series-A and Series-B companies whose initial pitch material is publicly available, run the panel on those pitches, and measure whether the verdict distribution correlates with what the market actually decided. This is the single most important experiment for the system to be defensible: without it, the panel is just three plausible-sounding voices.

2. **Reward modeling from real VC transcripts.** A curated corpus of recorded pitch sessions (consented Y Combinator demo days, France Digitale founder interviews, public investor podcasts) annotated with conviction trajectories. Train a small reward model on these trajectories to fine-tune the agents toward question patterns that actual experienced investors use, rather than the patterns a base LLM extrapolates from its training data.

3. **Multimodal extension.** Voice-based detection of hesitation, confidence and contradiction in the founder's responses. Treat paralinguistic features as additional conditioning for the agents, the way an in-person investor implicitly reads delivery.

4. **Inter-agent dynamics as a small MARL system.** The current set-up is a fixed prompt-conditioned policy. A natural next step is to study what happens when the agents have a small budget of moves (challenge, defer, change-conviction) and learn to coordinate or disagree to maximise an external scoring rule. This frames the panel as a controlled multi-agent reinforcement learning problem and lets us probe questions about agent diversity, collusion, and consensus formation that the prompt-only design cannot.

5. **Failure-mode auditing.** Adversarial pitches (deliberately bad metrics dressed up as good narrative; deliberately good fundamentals presented poorly; sector mismatches against agent specialism). Measure how robust each persona is and which patterns systematically fool the panel.

## Selected real session

A pitch from PetPal (a rescue-animal adoption marketplace) raising €200K at €1.67M valuation. Final score 59 / 100, one in, two out. Visionary went all-in (10/10 conviction, counter-offered 10 % equity instead of 12 %). Growth Hacker started impressed and reversed mid-session on cross-border scalability concerns raised by another agent. The reversal emerged from the multi-agent debate rather than from a scripted decision rule, which is the property the system is designed to preserve: any agent can vote out, including Visionary, so a founder can walk away with zero investors if the pitch does not hold up.
