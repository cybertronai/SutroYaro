# Example: exp_dmc_optimize (Phase 1 -> Phase 2)

Real walkthrough of a completed experiment, mirroring the two-phase protocol
in LAB.md. Source artifacts live at `docs/findings/exp_dmc_optimize.md` and
`results/exp_dmc_optimize/results.json`.

## Hypothesis

If we reduce KM influence samples from 5 to 1 (parity influence is binary,
never fractional) and reuse a single buffer in place, DMC drops below the
GF2 baseline of 8,607.

## Config

```python
n_bits=20, k_sparse=3, seed=42
# robustness: seeds 43, 44, 45, 46
method="km"   # plus variants A-E described in findings doc
```

## Phase 1: `results/exp_dmc_optimize/results.json`

Raw numbers only, no interpretation:

```json
{
  "experiment": "exp_dmc_optimize",
  "issue": "#22",
  "config": {"n_bits": 20, "k_sparse": 3, "seed": 42},
  "baseline_gf2_dmc": 8607.4,
  "results": {
    "baseline_gf2":  {"accuracy": 1.0, "ard": 420.0, "dmc":  8607.4, "total_floats":  860},
    "baseline_km":   {"accuracy": 1.0, "ard":  91.8, "dmc": 20632.5, "total_floats": 4420},
    "km_min":        {"accuracy": 1.0, "ard":  20.0, "dmc":  3578.0, "total_floats": 1600},
    "km_inplace":    {"accuracy": 1.0, "ard":  30.0, "dmc":  4319.0, "total_floats": 1200}
  }
}
```

## Phase 2: `docs/findings/exp_dmc_optimize.md`

Status: SUCCESS. KM-min (1 sample per bit) is the new DMC leader at 3,578
(58 percent below the GF2 baseline). All three optimized variants hit 100
percent accuracy across 5 seeds.

## Checklist trace

- [x] DISCOVERIES.md read (GF2 was the prior DMC leader)
- [x] One variable changed: sample count 5 -> 1
- [x] Seeds recorded: 42, 43, 44, 45, 46
- [x] results.json saved with config + environment
- [x] Findings doc written with Status: SUCCESS
- [x] log.jsonl updated with class: WIN
- [x] DISCOVERIES.md updated with new best DMC
