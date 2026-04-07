# ASI-Evolve Memory Architecture Analysis
**Agent**: Kimi 2.5 | **Role**: Systems Architect  
**Date**: 2026-04-07  
**Source**: ASI-Evolve paper (arXiv:2603.29640) | ByteDMD metric analysis

## Executive Summary

ASI-Evolve's framework provides a blueprint for structuring SutroYaro's experimental memory system. The paper's "Cognition Base" and "Analyzer" components map cleanly onto our existing markdown infrastructure. This analysis details how to formalize these mappings and strengthen our system against the specific constraint of **spatial locality** using the **ByteDMD metric** (our canonical energy proxy).

**Important**: SutroYaro has transitioned to **ByteDMD** as the primary energy efficiency metric. DMC (Data Movement Complexity) and ARD (Average Reuse Distance) are **deprecated** and retained only for historical comparison. ByteDMD provides byte-level granularity via LRU stack modeling: $C = \sum_{b \in bytes} \lceil\sqrt{D(b)}\rceil$.

## ASI-Evolve Architecture Overview

ASI-Evolve operates on a **learn-design-experiment-analyze** cycle with four core modules:

1. **Cognition Base (C)**: Structured repository of task-relevant knowledge (heuristics, pitfalls, design principles)
2. **Database (𝒟)**: Persistent store of experimental nodes (motivation, code, results, analysis, score, metadata)
3. **Researcher**: Generates candidate programs conditioned on sampled context + retrieved cognition
4. **Analyzer**: Distills complex multi-dimensional experimental outputs into compact, decision-oriented reports
5. **Engineer**: Executes experiments and produces evaluation signals

## Mapping to SutroYaro Infrastructure

### Cognition Base ↔ DISCOVERIES.md

**ASI-Evolve's Cognition Base** injects human priors via embedding-based retrieval. It contains:
- Domain literature (~150 papers in their architecture design task)
- Known pitfalls and failure modes
- Design principles and heuristics

**SutroYaro's DISCOVERIES.md** already serves this function:
- **Proven Facts**: 150+ bullet points across hyperparameters, grokking, training variants
- **Failed Approaches**: Hebbian (~50%), Predictive Coding (~55%), Equilibrium Propagation (~60%)
- **Open Questions**: Q7, Q11-Q13 prioritized by impact

**Gap**: DISCOVERIES.md is text-only; ASI-Evolve uses embedding-indexed retrieval. For agent autonomy, we could add semantic tags or maintain a parallel vector index.

**Critical Migration Needed**: DISCOVERIES.md currently references DMC/ARD values extensively. These need conversion to ByteDMD or clear deprecation markers to prevent agents from optimizing against obsolete metrics.

### Analyzer ↔ findings/ + DISCOVERIES.md Updates

**ASI-Evolve's Analyzer** receives full experimental output (logs, metrics, traces) and produces:
- Structured analysis reports
- Actionable insights written back to database
- Diagnosis of why experiments succeeded/failed

**SutroYaro's current flow**:
1. Experiment runs → `results/{exp_name}/results.json` (raw metrics)
2. Findings written to `findings/{exp_name}.md` (analysis per LAB.md template)
3. If answering open question → update DISCOVERIES.md

**Alignment**: Our two-phase protocol (Phase 1: evidence, Phase 2: interpretation) matches ASI-Evolve's evidence → analysis flow. The `findings/_template.md` structure captures the Analyzer's output format.

### Database ↔ research/log.jsonl

**ASI-Evolve's Database** stores nodes with:
- Researcher motivation
- Generated program (code)
- Structured results
- Analysis report
- Metadata (runtime, success flag)

**SutroYaro's research/log.jsonl** is append-only JSON lines with:
- Experiment ID, timestamp, hypothesis
- Config, results (accuracy, ByteDMD, time)
- Classification (WIN/LOSS/INVALID/INCONCLUSIVE)
- Researcher identification

**Gap**: We don't currently store full analysis reports in the log—only metrics and classification. The findings/ directory serves this archival purpose but isn't machine-readable for retrieval.

## ByteDMD: The Canonical Metric

The prompt emphasizes preventing agents from proposing algorithms that fail the **spatial locality constraint** modeled by ByteDMD. Unlike coarse-grained DMC, ByteDMD operates at the byte level.

### ByteDMD Model

ByteDMD (Data Movement Distance) models memory as an LRU stack where reading a byte at depth $d$ costs $\lceil\sqrt{d}\rceil$. This penalizes non-local reads that thrash cache.

From the ByteDMD repo:
- **Computation is free; only reads incur cost**
- **Multi-byte scalars treated as contiguous blocks**
- **Repeated inputs charged per occurrence** (e.g., `a + a` reads `a` twice)
- **Cost formula**: $C = \sum_{b \in bytes} \lceil\sqrt{D(b)}\rceil$

**Key distinction from DMC**: ByteDMD tracks individual byte accesses rather than abstract "data movement events," providing finer granularity and harder-to-game measurement.

### Metric Deprecation Status

| Metric | Status | Notes |
|--------|--------|-------|
| **ByteDMD** | **CURRENT** | Byte-level LRU stack model, canonical for all new experiments |
| **DMC** | **DEPRECATED** | Replaced by ByteDMD; historical values retained for comparison only |
| **ARD** | **DEPRECATED** | Averaged reuse distance; superseded by ByteDMD's sqrt-penalty model |

### Legacy DMC Values (For Reference Only)

DISCOVERIES.md currently tracks deprecated DMC rankings. These illustrate spatial locality failures but should not guide new algorithm design:

| Method | ByteDMD (TBD) | Legacy DMC (deprecated) | Spatial Locality |
|--------|---------------|-------------------------|------------------|
| KM-min (1 sample) | **TBD** | 3,578 | **Excellent** (single pass, small working set) |
| GF2 (harness) | **TBD** | 8,607 | **Good** (coarse tracking undercounts true cost) |
| KM (5 samples) | **TBD** | 20,633 | **Good** |
| SGD | **TBD** | 1,278,460 | **Poor** (repeated weight matrix reads) |
| Fourier | **TBD** | 78,140,662,852 | **Terrible** (streaming over data for each subset) |

**Note**: ByteDMD values are marked as "TBD" because the experiments need re-running with the ByteDMD wrapper. The ByteDMD repo shows DMC was undercounting by up to 22× in some cases.

## Recommendations for Cognition Base Structure

To prevent agents from proposing algorithms with poor spatial locality, extend DISCOVERIES.md (or create `docs/research/spatial-locality-rules.md`) with explicit tagging:

### 1. Tag Each Method with ByteDMD Profile

```markdown
## Method: Fourier/Walsh-Hadamard
**ByteDMD**: TBD (expected: very high due to O(C(n,k)) streaming)  
**Legacy DMC**: 78B (deprecated) | **Legacy ARD**: 11.9M (deprecated)  
**Spatial Locality**: FAIL - streaming over dataset for each subset  
**Pattern**: O(C(n,k)) dataset reads, no weight reuse  
**Verdict**: Avoid for energy-efficient training despite fast wall time

## Method: GF(2) Gaussian Elimination  
**ByteDMD**: TBD (expected: low due to small working set)  
**Legacy DMC**: 8,607 (deprecated) | **Legacy ARD**: 420 (deprecated)  
**Spatial Locality**: PASS - single-pass over small sample set  
**Pattern**: O(n) samples, O(n³) compute, minimal data movement  
**Verdict**: Preferred for exact solutions
```

### 2. Add "Anti-Patterns" Section

Explicitly document patterns that trigger poor ByteDMD scores:

```markdown
### Anti-Pattern: Per-Subset Data Streaming
**Example**: Fourier checking all C(n,k) subsets by reading full dataset per subset  
**ByteDMD Impact**: Extreme cost due to repeated full-dataset reads at increasing stack depths  
**Legacy DMC Impact**: 9M× worse than cached approach  
**Prevention**: Cache intermediate results or use single-pass algorithms

### Anti-Pattern: Repeated Weight Matrix Access
**Example**: SGD reading W1 (80KB) every forward/backward pass  
**ByteDMD Impact**: High cost from repeated reads of large matrices  
**Legacy ARD Impact**: ~17K despite L2 cache fitting working set  
**Prevention**: Per-layer updates reduce W1 reads by 3.8%
```

### 3. Encode ByteDMD-Specific Heuristics in Cognition

From ByteDMD paper and SutroYaro findings:

1. **Prefer algorithms with small working sets** (fit in L1/L2 cache)
2. **Prefer single-pass over multi-pass** on large datasets
3. **Prefer compute-heavy over data-heavy** (computation is free in ByteDMD model)
4. **Cache intermediate results** rather than recompute
5. **Avoid O(C(n,k)) enumeration** when k>3 (Fourier scales poorly in both wall time and ByteDMD)

## Analyzer Workflow Integration

The Analyzer should explicitly check spatial locality using ByteDMD and tag findings:

### Current Findings Template (LAB.md)

```markdown
## Results
| Metric | Value |
|--------|-------|
| Best test accuracy | |
| Epochs to >90% | |
| Wall time | |
| Weighted ARD | |  # DEPRECATED
```

### Proposed Extension (ByteDMD-Centric)

```markdown
## Results
| Metric | Value |
|--------|-------|
| Best test accuracy | |
| Epochs to >90% | |
| Wall time | |
| **ByteDMD** | |  # PRIMARY METRIC
| Legacy DMC (deprecated) | |  # For historical comparison only
| Legacy ARD (deprecated) | |  # For historical comparison only
| Spatial Locality Verdict | PASS/FAIL |

## Spatial Locality Analysis (ByteDMD Model)
- **Working set size**: [e.g., 80KB fits in L2]
- **Cache behavior**: [e.g., L2 eliminates all misses]
- **ByteDMD breakdown**: [e.g., reads from depth 1-10: 60%, depth 11-100: 30%, depth 100+: 10%]
- **Anti-patterns present**: [e.g., per-subset streaming]
- **ByteDMD optimization opportunities**: [if any]
```

## Database Schema Enhancement

Extend `research/log.jsonl` schema to support ByteDMD and spatial locality:

```json
{
  "experiment_id": "exp_fourier_v2",
  "metrics": {
    "bytedmd": {
      "total_cost": 78140662852,
      "bytes_accessed": 23961000,
      "locality_verdict": "FAIL",
      "anti_patterns": ["per_subset_streaming", "repeated_full_dataset_reads"]
    },
    "deprecated": {
      "dmc": 78140662852,
      "ard": 11980500,
      "note": "Retained for historical comparison only"
    }
  }
}
```

This enables the Researcher (agent) to filter proposals by:
- `metrics.bytedmd.locality_verdict == "PASS"` for energy-efficient tracks
- `anti_patterns` intersection check before proposing variants

## Migration Path: DMC/ARD → ByteDMD

**Immediate Actions**:
1. **Update DISCOVERIES.md**: Add deprecation notices to all DMC/ARD references
2. **Re-run key experiments** with ByteDMD wrapper to establish baselines
3. **Update LAB.md template** to prioritize ByteDMD over deprecated metrics

**Qwen's Task**: Create PR updating all repo docs to reflect ByteDMD as canonical metric

## Summary

ASI-Evolve's architecture validates SutroYaro's existing design:
- **DISCOVERIES.md** = Cognition Base (human priors)
- **findings/** = Analyzer output (structured insights)
- **research/log.jsonl** = Database (experimental memory)

**To strengthen against ByteDMD constraint**:
1. **Migrate to ByteDMD**: Mark DMC/ARD as deprecated in all docs
2. **Re-establish baselines**: Run key experiments with ByteDMD wrapper
3. **Tag methods with ByteDMD profiles** in Cognition Base
4. **Document anti-patterns** that cause poor ByteDMD scores
5. **Extend findings template** with ByteDMD-centric spatial locality analysis
6. **Enhance log schema** with structured ByteDMD metadata

This ensures agents optimize against the **correct** energy metric (ByteDMD) rather than deprecated proxies.

## Files

- Analysis: `docs/research/asi-evolve/kimi_asi_memory.md` (this file)
- Source Paper: https://arxiv.org/abs/2603.29640
- ByteDMD Metric (canonical): https://github.com/cybertronai/ByteDMD
- SutroYaro Context: DISCOVERIES.md (needs migration), CLAUDE.md, LAB.md, AGENT.md
