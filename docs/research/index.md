# Research

Deep research notes and literature review for the Sutro Group.

## Topics

- [x] Sparse parity learning theory — [literature review](sparse-parity-literature.md)
- [x] Average Reuse Distance -- theory and measurement
- [ ] Forward-Forward algorithm (Hinton 2022) on harder instances
- [ ] Energy-efficient training at scale (n=100, k=5)
- [ ] Sign SGD (Kou et al. 2024) implementation
- [ ] Cache-aware neural network training
- [ ] Local learning rules vs backpropagation

## Key Papers

| Paper | Year | Relevance | Link |
|-------|------|-----------|------|
| Hidden Progress in Deep Learning (Barak et al.) | 2022 | SGD learns sparse parity via hidden Fourier gap | [arxiv](https://arxiv.org/abs/2207.08799) |
| Matching SQ Lower Bound with Sign SGD (Kou et al.) | 2024 | Theoretically optimal sparse parity solver | [arxiv](https://arxiv.org/abs/2404.12376) |
| A Tale of Two Circuits (Merrill et al.) | 2023 | Grokking = sparse vs dense subnetwork competition | [arxiv](https://arxiv.org/abs/2303.11873) |
| GrokFast (Lee et al.) | 2024 | EMA gradient filter accelerates grokking | [github](https://github.com/ironjr/grokfast) |
| Feature Learning Dynamics under Grokking | 2024 | NTK eigenfunctions align with secret indices | [openreview](https://openreview.net/forum?id=gciHssAM8A) |
| Bill Daly - Energy in GPUs | 2024 | Memory cost dominates energy | [YouTube](https://youtu.be/rsxCZAE8QNA?si=8-kIJ1MuhxChRLgW&t=2457) |

## Other Resources

| Resource | Type | Link |
|----------|------|------|
| Fitting Larger Networks into Memory | Article | [Medium](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9) |
| Sparse Parity background | Notebook | [NotebookLM](https://notebooklm.google.com/notebook/3eb7b53e-168f-409c-a679-2fb009119e2e) |
| Sparse Parity Optimization | Slides | [PDF](https://drive.google.com/file/d/1nC5KbckpwLjeGynuS4wPntBoClfDHyqY/view) |
| Hinton's Forward-Forward | Paper + Discussion | [Group notes](https://docs.google.com/document/d/1IdXRUhPRoWt8xLH1Y6iRWRx1g9-gbotFiiAnVixJYZY/edit?tab=t.0) |
| ARD Brainstorming | Gemini session | [Session](https://gemini.google.com/share/c99ec90874da) |
| parity-nn (minimal codebase) | GitHub | [Tsili42/parity-nn](https://github.com/Tsili42/parity-nn) |

## Concepts

### Average Reuse Distance (ARD)

Proxy metric for energy efficiency. When ARD is small, data stays in fast, energy-efficient cache. When ARD is large, data must be fetched from expensive external memory (HBM).

### The Giraffe Nerve Analogy

Backpropagation is like the recurrent laryngeal nerve in giraffes -- it works but is wildly inefficient because of the global memory access pattern. The brain uses ~20 Watts with local update rules. We want to find the AI equivalent.
