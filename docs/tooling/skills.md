# Skills Reference

Skills are reusable workflow templates for Claude Code. They enforce discipline on specific types of work — you invoke them with `/skill-name` or Claude Code invokes them automatically when relevant.

## Skills We Use

### Anti-slop Guide

**Invoked**: `/anti-slop-guide` or automatically when writing prose

Detects and removes AI writing patterns: overused vocabulary (delve, tapestry, landscape), formulaic structures (binary contrasts, rule of three), throat-clearing openers, business jargon. See the [full guide](anti-slop-guide.md).

We ran an anti-slop pass on all 32 MkDocs pages after initial generation. The difference was significant — pages went from sounding like ChatGPT marketing copy to reading like research notes.

### Brainstorming

**Invoked**: Automatically before any creative work (features, components, behavior changes)

Explores intent, requirements, and design before implementation. Prevents jumping straight to code.

### Systematic Debugging

**Invoked**: When encountering any bug, test failure, or unexpected behavior

Forces root-cause analysis before proposing fixes. Prevents the "try random things until it works" pattern.

### Test-Driven Development

**Invoked**: Before writing implementation code

Write tests first, then implementation. Keeps experiments honest — if the test passes, the experiment succeeded.

### Writing Plans / Executing Plans

**Invoked**: For multi-step tasks before touching code

Write a plan, get approval, then execute with review checkpoints. Used for the experiment pipeline design.

### Parallel Agent Dispatch

**Invoked**: When facing 2+ independent tasks

Spawns multiple Claude Code agents working in parallel on independent tasks. Used to run multiple experiments simultaneously.

### Verification Before Completion

**Invoked**: Before claiming work is done

Requires running verification commands and confirming output before making success claims. Prevents "it should work" without evidence.

### Code Review

**Invoked**: After completing a major feature or step

Reviews implementation against the original plan and coding standards.

## Skills Worth Building

These don't exist yet but would fit the Sutro Group workflow:

### Research Sprint

A skill that enforces the research loop from [prompting strategies](../findings/prompting-strategies.md):

1. Literature search on exact problem
2. Compare config against published baselines
3. Ranked experiment plan
4. Execute one experiment at a time
5. Analyze failures

### Energy Audit

A skill that runs ARD analysis on a given experiment script:

1. Run the experiment
2. Measure ARD with CacheTracker
3. Compare against baseline
4. Identify the top-3 memory access bottlenecks
5. Suggest improvements

### Google Docs Sync

A skill that wraps `sync_google_docs.py`:

1. Pull latest docs
2. Diff against previous versions
3. Update cross-references
4. Rebuild MkDocs site

## How to Create a Skill

Skills live in `~/.claude/skills/` as directories with a markdown file. The file has YAML frontmatter (name, description) and the skill content.

```
~/.claude/skills/my-skill/
  my-skill.md    # Skill definition with frontmatter
```

Invoke with `/my-skill` in Claude Code.
