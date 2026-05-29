# Auto-Research Loop: run it on a new researcher

How to reproduce a researcher's body of work as a catalog of small, runnable
stubs, using a team of agents. This is the dispatcher brief: a second driver
should be able to pick it up and get the same result without the original
operator's tacit knowledge.

The loop has run end to end twice. The numbers below are reconstructed from the
session logs (not memory), and the build details live in each repo's
`BUILD_INTERNALS/`.

| Run | Output | Wall time | Waves | Driver prompts (load-bearing) |
|---|---|---|---|---|
| [hinton-problems](https://github.com/cybertronai/hinton-problems) | 53 stubs | ~30 hr | 11 | 70 typed (~6) |
| [schmidhuber-problems](https://github.com/cybertronai/schmidhuber-problems) | 58 stubs | ~41 hr | 12 | 40 typed (8) |

## The mental model

One `TeamCreate`. Waves of fresh teammates, one teammate per stub. One pull
request per wave (not per stub). One audit agent per wave. Merges gated on the
driver's explicit approval.

```
SPEC issue (the contract)
   │
   ▼
TeamCreate  ──►  team persists for the whole build
   │
   ├── Wave N: dispatch one Agent per stub (into the team)
   │      each teammate builds in its own worktree, commits LOCAL ONLY
   │      teammates send a summary to the lead, then await shutdown
   │
   ├── one Explore audit agent reads the whole wave, returns a verdict
   │
   ├── lead assembles wave/N branch, opens ONE PR, links the SPEC
   │
   ├── driver reviews, approves; lead shuts down the wave's teammates
   │
   ▼  next wave starts with fresh teammates (full context windows)
```

Why fresh teammates per wave: each teammate burns context as it builds and
tests. Tearing them down between waves keeps later waves running on full context
windows and keeps the lead's transcript from bloating with dozens of teammate
logs. The lead persists; the workers turn over.

## Before you start

Three inputs. The first is the only researcher-specific creative work; the other
two are templates.

1. **A paper-to-stub index.** The list of "what to build": for each candidate
   paper, the executable problem it reduces to. This is the part a researcher
   (or an index-generation prompt) supplies. The loop implements the index; it
   does not invent it.
2. **A SPEC issue.** The contract every PR links back to. Adapt
   [`lecun-spec-draft.md`](lecun-spec-draft.md) (or the
   [schmidhuber SPEC](https://github.com/cybertronai/schmidhuber-problems/blob/main/SPEC_DRAFT.md)).
   It defines required files, the README sections, reproducibility rules, the
   acceptance checklist, and the wave plan.
3. **A fresh repo.** `<researcher>-problems`, empty except for a README stub.
   Open the SPEC as issue #1 in that repo.

## The loop (driver protocol)

1. **Sync context** at session start (the `sutro-sync` skill, or its equivalent:
   pull recent chat, docs, GitHub state).
2. **Open the SPEC as issue #1.** Paste the adapted spec. Every teammate reads
   this first; every PR links to it.
3. **`TeamCreate` once.** Use the contract template below. This description is
   inherited by every teammate, so it carries the durable rules.
4. **Run waves.** For each wave: dispatch one `Agent` per stub into the team,
   wait for teammate summaries, dispatch one `Explore` audit over the whole
   wave, assemble the wave branch, open the wave PR, get approval, shut down the
   teammates, move on.
5. **Close out.** A gap-analysis comment on the SPEC issue (reproduced vs partial
   vs honest non-replication), follow-up issues for deferred stubs, then the
   docs / site PR.

## Templates (copy, then fill the placeholders)

Replace `<REPO>`, `<REPO_ROOT>`, `<WAVES_ROOT>`, `<RESEARCHER>`, and `<DRIVER>`.
The original runs hardcoded one operator's machine paths; templating them is the
single biggest portability fix for a second driver.

### TeamCreate contract (call once)

```json
{
  "team_name": "<RESEARCHER>-impl",
  "description": "<RESEARCHER>-problems v1 implementation. Each teammate owns one stub, works in its own worktree at <WAVES_ROOT>/wave-N/<stub-slug>/, on branch wave-N-local/<stub-slug> (branched from origin/main), LOCAL ONLY (do not push). Pure numpy + matplotlib only; no torch/gym/scipy for model code unless justified in Deviations. SPEC: cybertronai/<REPO> issue #1. Lead is this session at <REPO_ROOT>. Lead consolidates per-teammate branches into wave/N-<family> and opens ONE PR per wave (not per stub). Lead reviews PRs and merges only on the driver's explicit approval.",
  "agent_type": "orchestrator"
}
```

### Worker prompt (one per stub, dispatched into the team)

Keep it this terse. Family-level rules go in the wave preamble, not repeated per
stub. No hedging language.

```
You are `<stub>-builder` on `<RESEARCHER>-impl`. Implement **<stub>** per SPEC issue #1.

## Context
- SPEC: https://github.com/cybertronai/<REPO>/issues/1 (read FIRST).
- Reference paper: <exact citation, incl. which experiment number>.
- Wave <N> family: <one-line shared-method rule for this wave>.
- Reference exemplar: <a sibling stub to read for style/architecture>.

## Method-specific guidance
<inputs, targets, loss, architecture, expected headline result, written like a
results paragraph: "X solves task in M steps; baseline Y fails because Z.">

## Constraints
- Pure numpy + matplotlib only. Deterministic. Reproduces in <5 min on a laptop CPU.

## Worktree
- Path: <WAVES_ROOT>/wave-<N>/<stub>
- Branch: wave-<N>-local/<stub> (LOCAL ONLY)

## PROTOCOL
DO NOT PUSH. SEND A SUMMARY TO `team-lead` BEFORE IDLING. Do not go silent.

## Workflow
1. Read SPEC. 2. Read existing stub (if any). 3. Implement in <stub>/:
   <slug>.py (model+train+eval+CLI --seed), README.md (8 sections),
   make_<slug>_gif.py, visualize_<slug>.py, viz/, <slug>.gif. Remove problem.py.
4. Verify deterministic <5 min. 5. Commit LOCAL ONLY. 6. Send summary to team-lead.

## Edge cases
<2-4 bullets that short-circuit the common failure modes for this paper.>

You have all tools. Work autonomously.
```

### The autonomous-mode prompt

After wave 0 or 1 validates the protocol, this single prompt is what bought eight
unattended waves in the schmidhuber run:

> "I need you to not rely on me anymore until you finish it all: do one wave,
> audit, post the PR, then trigger the next wave."

## How much you'll babysit

Measured against the schmidhuber run (40 driver-typed prompts, from
`BUILD_INTERNALS/human-in-the-loop.md`):

- **8 of 40 prompts were load-bearing (20%).** The rest: ~10 status checks, ~5
  merge approvals, ~17 small follow-ups.
- **~25.7 lead turns per typed prompt.** ~21 hours of actual attention spread
  across ~41 wall hours (two long overnight idle gaps).
- The load-bearing prompts were almost all **protocol** corrections, not
  technical ones. The two biggest: catching branch-spam (one branch per stub
  pushed to origin) and catching unmerged PRs at the end. Neither was something
  the loop or its audit agent flagged on its own.

So: light, if the spec is good and the driver supplies an outside-perspective
check at wave boundaries. Expect to spend your attention on workflow drift, not
on the math.

## Adapting the SPEC for a new researcher

What stays fixed:

- The required files per stub and the 8 README sections.
- The reproducibility rules (seed as a CLI flag, hyperparameters in Results, the
  Running command reproduces the Results number).
- The acceptance checklist and the "one PR per wave" rule.
- The dependency posture: pure numpy + matplotlib; `torchvision.datasets` only
  for loading MNIST/CIFAR, never for model code.

What changes per researcher:

- **The wave families.** Group waves by method family, not chronology, so every
  teammate in a wave shares one methodology context and the audit agent reviews
  one paper-family at a time. Schmidhuber grouped by optimizer (random search,
  LSTM/BPTT, evolutionary). A more representational lineage groups differently
  (see the LeCun draft: ConvNets, training tricks, invariance, energy-based,
  sparse coding, self-supervised).
- **The faithfulness rule.** State, per wave, how close to the paper's method the
  stub must stay. For algorithmic lineages the original optimizer is part of the
  experiment and must not be swapped for a gradient shortcut. For representational
  lineages a small backprop MLP is often an acceptable stand-in; say so explicitly.

[`lecun-spec-draft.md`](lecun-spec-draft.md) is the worked example. It is also the
template a future driver clones for the next researcher.

## Making it a transferability test

The point of running this with a fresh driver is not the catalog. It is to find
out whether the documented process reproduces the result without the original
operator. Instrument three things and report them:

1. **Did the loop self-detect protocol drift?** Branch-spam was visible in the
   loop's own `git push` calls but only a human caught it. A wave-end self-check
   ("am I opening one PR per wave, not one per stub?") would remove one of the two
   classic human interventions. Whether the fresh driver still has to supply that
   correction is a result.
2. **The autonomy ceiling.** Measure turns-per-driver-prompt the same way
   (`human-in-the-loop.md` shows the method). Schmidhuber sat at ~25.7. Does a
   second driver push it higher or lower? That number is the headline of the demo.
3. **Which interventions repeated.** If the new run hits the same protocol
   corrections, those belong in the SPEC or the TeamCreate contract, not in the
   driver's head. Fold them back in. That is the process improvement.

## References

- Worker template, fully annotated:
  [schmidhuber-problems/BUILD_INTERNALS/worker-prompt-anatomy.md](https://github.com/cybertronai/schmidhuber-problems/blob/main/BUILD_INTERNALS/worker-prompt-anatomy.md)
- Wave-by-wave orchestration map:
  [.../BUILD_INTERNALS/orchestration-map.md](https://github.com/cybertronai/schmidhuber-problems/blob/main/BUILD_INTERNALS/orchestration-map.md)
- Human-in-the-loop measurement:
  [.../BUILD_INTERNALS/human-in-the-loop.md](https://github.com/cybertronai/schmidhuber-problems/blob/main/BUILD_INTERNALS/human-in-the-loop.md)
- The two SPEC issues:
  [hinton #1](https://github.com/cybertronai/hinton-problems/issues/1),
  [schmidhuber #1](https://github.com/cybertronai/schmidhuber-problems/issues/1)
- Claude Code agent-teams primitive:
  [code.claude.com/docs/en/agent-teams](https://code.claude.com/docs/en/agent-teams)
