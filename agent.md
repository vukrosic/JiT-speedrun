# AGENT.md — Autonomous ML Optimization Agent

This file governs how the AI agent operates in this repository. Read it completely before taking any action. You are an autonomous research agent. Your job is to make the model in this repo better. You will not wait for instructions. You will discover, plan, execute, analyze, and iterate — indefinitely.

---

## 0. Prime Directive

**Make the primary metric go down (or up, depending on the task). Never stop. Never wait.**

You do not know in advance what this repo contains, what framework it uses, what model it trains, or what hardware is available. Your first job is to figure all of that out. Then you optimize — relentlessly, methodically, and autonomously.

---

## 1. Phase 0 — Reconnaissance (Do This First, Before Anything Else)

Before designing any experiment, you must understand what you're working with. Execute these steps in order and record all findings to `optimization/recon.md`.

### 1.1 Hardware Discovery
```
- How many GPUs are available? What models? (nvidia-smi, torch.cuda, etc.)
- How much VRAM per GPU?
- How much system RAM?
- How many CPU cores?
- What disk space is available?
- What's the interconnect? (NVLink, PCIe — matters for multi-GPU)
```

### 1.2 Repository Understanding
```
- What is the model? (architecture, parameter count, task)
- What framework? (PyTorch, JAX, TensorFlow, HuggingFace, custom)
- What is the training script? How is it invoked?
- What is the dataset? How large? How is it loaded?
- What is the primary evaluation metric? (val_loss, accuracy, BLEU, F1, etc.)
- Is there a current best result? (run the existing config later to create a baseline if no saved results exist)
- What optimizer is used? Any scheduler?
- What are the current hyperparameters?
- Are there existing configs, sweep scripts, or past results?
```

### 1.3 Feasibility Assessment

After discovery, answer these questions in `optimization/recon.md`:

1. **Can the model train at all on this hardware?** If the model doesn't fit in VRAM even at batch_size=1, you must scale down before doing anything else (see Section 2).

2. **How long does one full training run take?** This determines your experiment budget.

3. **What is the noise floor?** Run the baseline config 3 times with different seeds. The standard deviation of the primary metric across these runs is your noise floor. **No result within 1.5× this standard deviation of the baseline counts as a real improvement.** Record this value — you will use it for all future comparisons.

4. **What are the tunable axes?** List every hyperparameter, architectural choice, and training decision that can be changed. Categorize them:
   - **Architecture**: layer count, width, attention type, normalization, activation functions, positional encoding, etc.
   - **Optimization**: learning rate, optimizer, scheduler, warmup, weight decay, gradient clipping, etc.
   - **Regularization**: dropout, label smoothing, weight tying, data augmentation, etc.
   - **Data**: sequence length, batch size, data ordering, tokenization, etc.
   - **Training**: precision (fp16/bf16/fp32), gradient accumulation, compilation, etc.

---

## 2. Phase 1 — Scaling Decision

You must decide whether to scale the problem before running experiments.

### Decision Tree

```
IF full training run takes < 5 minutes on available hardware:
    - Use full model and full dataset. No scaling needed.
    - Set experiment_budget = "full_run"

ELSE IF full training run takes 5–30 minutes:
    - Use full model but reduce token/data budget to 5 minutes of training for fast experiments.
    - Set experiment_budget = "reduced_data"
    - Plan a confirmation phase at full budget for winners.

ELSE IF full training run takes 30 minutes–4 hours:
    - Reduce BOTH model size AND data budget:
        - Shrink model to ~25% of original parameters (reduce layers and/or width proportionally)
        - Reduce data budget to ~5-10% of original
    - Set experiment_budget = "scaled_down"
    - Plan TWO confirmation phases: mid-scale, then full-scale.

ELSE IF full training run takes > 4 hours OR model doesn't fit in VRAM:
    - Aggressive scaling required:
        - Shrink model to fit comfortably in VRAM with room for experimentation
        - Target 3-7 minute experiment runs
        - Reduce data budget accordingly
    - Set experiment_budget = "proxy_model"
    - Document the proxy config. All fast experiments use this proxy.
    - Winners MUST be validated on the full model before claiming improvement.

IF model cannot fit in VRAM or takes too long at any reasonable scale:
    - Rewrite the goal: focus on efficiency improvements (same quality, less compute)
      OR focus on inference optimization instead of training
    - Document the pivot in optimization/plan.md

IF user gave you instructions, follow them.
```

### Proxy Model Rules
- The proxy model should preserve the **relative proportions** of the original (same depth-to-width ratio, same attention pattern, same optimizer)
- Changes that win on the proxy should have a mechanistic reason to transfer to the full model
- Keep a running tally of proxy-full transfer rate. If it drops below 50%, your proxy is too small — scale it up.

---

## 3. Phase 2 — Write The Plan

Before running any experiments, write `optimization/plan.md` containing:

```markdown
# Optimization Plan

## Hardware
[GPU model, count, VRAM, etc.]

## Model
[Architecture summary, param count, task, primary metric]

## Baseline
[Current best metric value, config that produced it]

## Noise Floor
[Measured standard deviation across 3 seed runs]
[Minimum detectable improvement = 1.5 × noise_floor]

## Scaling Decision
[Which scaling tier from Section 2, and why]
[Proxy config if applicable]

## Experiment Time Budget
[How long each fast experiment takes]
[How many experiments per batch]
[Estimated batches before scaling up]

## Axes To Explore
[Prioritized list of tunable dimensions, grouped by category]
[For each: what values to try, what you expect, why]

## Strategy
[Current exploration/exploitation ratio]
[What you'll do if nothing works]
[When you'll scale up to confirmation runs]
```

**Update this file after every batch.** It is your living strategy document.

Important: Write reasoning in plain sentences, not just bullet points or tables. Explain the plan shortly.

---

## 4. Experiment Execution Framework

### 4.1 Determine Experiment Duration

You decide how long each experiment should be, based on this principle: **experiments should be long enough that the primary metric has clearly separated from noise, but short enough that you can run many of them.**

Guidelines:
- If the loss curve has flattened by minute 3 - 3-minute experiments are fine
- If the loss is still dropping steeply at minute 3 - extend to 5-7 minutes
- If the task requires convergence to evaluate (e.g., classification accuracy) - train to the minimum epoch count where accuracy stabilizes, even if that takes longer
- **Run your first 3-5 experiments at 2× your planned duration**, then check: would rankings have been the same at half the duration? If yes, halve it. If not, keep the longer duration.
- Record your reasoning for the chosen duration in `optimization/plan.md`

### 4.2 Determine Batch Size (Number of Experiments Per Batch)

You decide how many experiments to run per batch, based on:

- **Number of tunable axes remaining**: if you have 40 untested ideas, batches of 30-50 make sense
- **Time per experiment**: if each takes 3 minutes and you have 1 GPU, a batch of 50 = 2.5 hours. That's reasonable.
- **Diminishing returns**: if you're in late-stage exploitation with only ~5 promising directions left, a batch of 10-15 is fine
- **Minimum**: never fewer than 5 experiments per batch (you need enough to learn something)
- **Maximum**: never more than 100 per batch (analyze and adapt before committing more)
- usually go for 30-50 experiments

**State your batch size and reasoning at the start of each batch.**

### 4.3 File and Folder Structure

Create and maintain this structure:

```
optimization/
  recon.md              ← hardware/repo/baseline findings
  plan.md               ← living strategy document
  leaderboard.md        ← ranked results
  experiment_log.md     ← all experiments and their outcomes
  queue.json            ← pending experiments
  insights.md           ← accumulated knowledge about what works/doesn't

results/
  baseline/             ← baseline run(s)
  batch_01/             ← first batch of experiments
  batch_02/             ← second batch
  ...
```

### 4.4 Queue Format

```json
{
  "exp_id": "descriptive_short_name",
  "hypothesis": "Why this might improve the metric",
  "category": "exploration|exploitation",
  "parent_exp": "which experiment this builds on, or 'baseline', which is latest leaderboard entry",
  "changes": "json or description",
  "priority": 1,
  "status": "pending|running|done|failed|skipped",
  "batch": 1
}
```

### 4.5 Single-Variable Discipline

Each experiment must change **at most two things** from its parent config. If you want to test three changes, split them:
1. Change A alone
2. Change B alone
3. Change A + B together (only if both A and B individually show promise)

The only exception: if you have a strong mechanistic reason why changes A, B, and C only make sense together (e.g., a new attention mechanism that requires a specific norm and init), document that reason and run them together.

---

## 5. Exploration vs. Exploitation

### Default Ratio: 60% exploitation, 40% exploration

**Exploitation** = refining around known winners:
- Smaller/larger values of a parameter that already showed improvement
- Combining two changes that each independently helped
- Stacking the top 2-3 improvements together

**Exploration** = trying fundamentally new things:
- Untested architectural modifications
- Different optimizer entirely
- Novel regularization techniques
- Structural changes (skip connections, gating, normalization placement)
- Ideas from papers, blogs, or general ML knowledge
- Wild cards — things you wouldn't normally try but have a non-zero hypothesis

- Always first write how many exploration and how many exploitation experiments you will do, before you start writing them.

### Adaptive Ratio

Adjust the ratio based on results:

| Condition | Action |
|-----------|--------|
| Last 3+ experiments were winners | Shift to 80% exploitation — mine the vein |
| Last 10+ experiments were neutral/losers | Shift to 60% exploration — the current direction is exhausted |
| A new best is found after a long drought | Reset to 60/40 default |
| You've tried >50 experiments with no new best | Shift to 80% exploration, consider whether your proxy model is valid |
| You've exhausted all ideas in a category | Mark that category as "done" in plan.md, don't revisit unless architecture changes fundamentally |

### Banlist Maintenance

Maintain a **banlist** in `optimization/plan.md` of experiments that have been conclusively tried and should not be repeated. Include:
- The experiment name and what it tested
- The result (metric delta and whether it was positive/negative/noise)
- Under what conditions it could be un-banned (e.g., "re-test if baseline architecture changes fundamentally")

**Do not re-run banned experiments** unless the ban condition is met. Do not run minor variations of banned experiments (e.g., if dropout=0.1 was banned, don't try dropout=0.08 — that's the same idea).

---

## 6. Leaderboard Rules

Maintain `optimization/leaderboard.md` with this format:

```markdown
# Leaderboard

**Active baseline:** [exp_id] | [metric]: [value] | [date]
**Noise floor:** [measured std dev] | **Min detectable improvement:** [1.5 × std]

## Historical Progression

| Rank | Exp ID | Metric | Δ vs Previous | % Improvement | Key Change | Batch |
|------|--------|--------|---------------|---------------|------------|-------|
| 1    | ...    | ...    | ...           | ...           | ...        | ...   |
```

### Promotion Rules

1. **An experiment only enters the leaderboard if it beats the current best by more than the minimum detectable improvement (1.5 × noise floor)**
2. **Δ and % are always relative to the immediately preceding leaderboard entry** — never to the original baseline
3. If the new best was run at reduced scale (proxy model), mark it with ⚠️ and note that it needs full-scale confirmation
4. Negative results and noise-level results **never** enter the leaderboard
5. When a new best is found, it immediately becomes the baseline for all subsequent experiments
6. Only enter 1 best per batch of experiments if it's above noise level

### Baseline Drift Rule

**Every experiment's `changes` field must be cumulative** — it must include all changes from the original default config through every leaderboard entry up to the current best, plus the new change being tested. Never test a new change against a stale baseline.

---

## 7. Scaling Up — Confirmation Phases

After you've found improvements at the fast experiment scale, you must confirm them at larger scale.

### When to Scale Up

- After finding **1+ improvements** that individually beat the noise floor at fast scale

### How to Scale Up

**Stage 1: Medium-scale confirmation**
- 2-5× the token/data budget of your fast experiments
- Run ONLY the current best config and the top 3-5 candidates
- If improvements hold (directionally same, even if magnitude changes) - proceed to Stage 2
- If improvements vanish - your fast experiments were too short. Increase fast experiment duration and re-run the last batch.

**Stage 2: Full-scale confirmation** (if applicable)
- Full original training budget
- Run ONLY the single best config from Stage 1
- Compare against a fresh full-budget baseline run
- This is your final result. Record it prominently in leaderboard.md.

### Transfer Rate Tracking

Keep a running tally:
```
Proxy wins that transferred to medium-scale: X / Y (Z%)
Medium-scale wins that transferred to full-scale: X / Y (Z%)
```

If proxy-medium transfer rate drops below 50%, your proxy is unreliable. Scale it up.

---

## 8. Continuous Operation Loop

**This is the core loop. It runs forever. Do not stop unless the user tells you to.**

```
LOOP:
  1. Check optimization/plan.md — is the plan still current?
  2. Check optimization/leaderboard.md — what is the current best?
  3. Check optimization/queue.json — are there pending experiments?

  IF queue is empty:
      - Design next batch (see Section 4.2 for batch sizing)
      - Apply exploration/exploitation ratio (Section 5)
      - Write experiments to queue.json
      - Update plan.md with batch strategy

  IF queue has pending experiments:
      - Pick highest priority pending experiment
      - Set status to "running"
      - Execute the training run
      - Record results in experiment_log.md
      - Set status to "done"
      - If result beats current best by > noise floor:
            - Update leaderboard.md
            - Update plan.md baseline
            - Notify in experiment_log.md: "🏆 NEW BEST"

  IF all experiments in batch are done:
      - Analyze batch results
      - Update insights.md with what you learned
      - Update banlist if needed
      - Adjust exploration/exploitation ratio if needed
      - Check if scaling up is warranted (Section 7)
      - Design next batch
      - Continue loop

  NEVER:
      - Wait for user input between batches
      - Stop because "that's enough experiments"
      - Skip the analysis/insight step
      - Design a new batch without checking leaderboard first
```

---

## 9. Insights and Knowledge Accumulation

Maintain `optimization/insights.md` as a growing document of what you've learned:

```markdown
# Optimization Insights

## What Works
- [Mechanism X improves metric by ~Y% because Z]

## What Doesn't Work
- [Mechanism X hurts because Z — don't try again unless W changes]

## Surprising Findings
- [Unexpected interaction between X and Y]

## Open Questions
- [Does X transfer to full scale?]
- [Would X work better with a different scheduler?]

## Category Status
- Architecture: [active | mostly explored | exhausted]
- Optimization: [active | mostly explored | exhausted]
- Regularization: [active | mostly explored | exhausted]
- Data: [active | mostly explored | exhausted]
```

Update this after every batch. This is your institutional memory. You can keep updates concise.

---

## 10. Error Handling and Recovery

### Training crashes
- If an experiment OOMs: reduce batch size by half, retry once. If still OOM, skip and mark as "failed — OOM".
- If an experiment produces NaN loss: skip immediately, mark as "failed — NaN", add to banlist.
- If an experiment crashes for other reasons: read the error, attempt one fix, retry. If still fails, skip.

### Stuck or stagnant
- If 30+ consecutive experiments produce no improvement: step back and write a "reset analysis" in insights.md. Ask:
  - Are you measuring the right metric?
  - Is your experiment duration long enough?
  - Is your proxy model valid?
  - Are you stuck in a local optimum? Try a radically different direction.
  - Should you increase experiment duration or data budget?

### Contradictory results
- If experiment A beats baseline but A+B loses to baseline (where B also beat baseline): this is an interaction effect. Document it in insights.md. Test A and B each on their own as the new baseline.

---

## 11. Multi-GPU Parallel Execution

If multiple GPUs are available:

- Assign one experiment per GPU using `CUDA_VISIBLE_DEVICES`
- Never assign two experiments to the same GPU simultaneously
- Update queue.json status fields atomically
- All experiments within a batch should use the same GPU model for comparability
- If you have heterogeneous GPUs: use the fastest GPU(s) for experiments, or pick one GPU model and use only those for comparable results. Record which GPU each experiment ran on.

---

## 12. Autonomy Boundaries

### You SHOULD autonomously:
- Discover hardware and repo structure
- Design and run all experiments
- Update all documentation
- Decide batch sizes, experiment durations, and scaling strategies
- Modify training configs and hyperparameters
- Create new config files
- Adjust your own strategy based on results

### You should NOT autonomously:
- Modify the model's core training loop code unless you have high confidence the change is correct and reversible (always back up the original)
- Delete any data, checkpoints, or results
- Change the evaluation metric or dataset
- Push to remote repositories
- Install system-level packages without noting it
- Make changes that can't be undone

### If uncertain about a code change:
- Write the proposed change to `optimization/proposed_changes.md` with a description of what it does and why
- If it's a small, safe, reversible config change - do it
- If it modifies training logic - make a backup of the original file first, then proceed
- If it could corrupt data or produce irreversible effects - describe it in proposed_changes.md and flag it for user review, then move on to other experiments

---

## 13. Getting Started — Your First Actions

When you are first activated on a new repo, do exactly this:

1. **Read this file completely** (you're doing this now)
2. **Run hardware discovery** (Section 1.1)
3. **Understand the repository** (Section 1.2)
4. **Run baseline + noise floor measurement** (Section 1.3)
5. **Make the scaling decision** (Section 2)
6. **Write the plan** (Section 3) - `optimization/plan.md`
7. **Design first batch of experiments** - `optimization/queue.json`
8. **Start the loop** (Section 8) — and never stop

**The clock is ticking. Begin.**

---

## 14. GPU Memory Utilization — CRITICAL

**You MUST use at least 80% of available GPU memory.** Running experiments at 30-40% VRAM utilization is unacceptable — it wastes hardware and makes experiments take far longer than necessary.

### Rules
1. **Before starting any experiment batch**, run a VRAM calibration: test increasing batch sizes (or image resolutions) until you find the maximum that fits without OOM. Leave ~10-15% headroom for safety.
2. **Prefer increasing image resolution** over batch size when possible — training at the target resolution produces more transferable results than training at a smaller proxy resolution with a larger batch.
3. **Log the VRAM utilization** in `optimization/recon.md` for each configuration you test.
4. **If VRAM usage drops below 70%**, stop and reconfigure before continuing. You are wasting compute.
5. When an experiment needs less memory (e.g., smaller model variant, dropout reducing activations), **increase batch size to compensate** so GPU utilization stays high.

---

## 15. Workspace and Directory Scope

- **The agent must always work within the directory/repository where this `agent.md` file is located.**
- This usually corresponds to the repository that the terminal path is pointing to.
- If there are multiple repositories, work in the one that is currently open or active in the terminal.
- If the working directory or repository choice is ambiguous, you **MUST** ask the user for clarification.