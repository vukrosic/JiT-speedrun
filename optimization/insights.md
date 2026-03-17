# Optimization Insights

## What Works
- **Smaller batch size (bs=128 vs 256)**: 9% improvement, same effective LR but 2x more gradient steps. Biggest single improvement since LR tuning.
- **Higher learning rates up to 1e-3**: Full LR curve mapped.
  - 2e-5: 0.2314 (worse than baseline)
  - 5e-5: 0.1923 (baseline)
  - 1e-4: 0.1675 (-12.9%)
  - 2e-4: 0.1479 (-23.1%)
  - 5e-4: 0.1353 (-29.6%)
  - 7e-4: 0.1322 (-2.3% vs 5e-4)
  - **1e-3: 0.1282 (-5.2% vs 5e-4) ← PEAK**
  - 2e-3: 0.1517 (+18.3% — reversal)
  - 3e-3: 0.1964 (near divergence)
  - 5e-3: 0.3277 (diverged)

## What Doesn't Work
- LR > 1e-3: sharp reversal, divergence above 2e-3
- Very low LR (2e-5): underfits
- Cosine LR schedule: uniformly worse than constant in 8-epoch runs
- warmup=0 with constant: spike causes instability
- warmup=3 with constant: diverged (NaN at 44s) — too many low-LR epochs then hard transition

## Surprising Findings
- Default LR (5e-5) was ~20x too low for bs=256 at 128px
- Optimal LR is 1e-3 (20x default). Very sharp cliff above that.
- Cosine decay actively hurts in short proxy runs — spending epochs at reduced LR costs more than it gains
- warmup=1 is the sweet spot; warmup=3 catastrophically diverges at blr=1e-3

## Open Questions
- bottleneck_dim=256 consistently near-threshold (-0.0017 to -0.0022). Would bottleneck_dim=384 or 512 push past noise floor?
- Would combining bottleneck_dim=256 + in_context_len=64 compound the small gains?
- Will results at 128px transfer to 256px?

## Banlist
- warmup=3 + constant LR at blr=1e-3: diverges (NaN)
- cosine schedule: consistently worse than constant in short runs
- in_context_start=0: worse than default (layer 4)
- in_context_start=2: worse than default
- bs=256 at 9 epochs: 30% worse than bs=128 at 8 epochs

## Category Status
- Optimization/LR: **exhausted** — peak at 1e-3 (bs=256), 2e-3 (bs=128)
- LR Schedule + Warmup: **exhausted** — constant + warmup=1 is best
- Weight Decay: **exhausted** — marginal, within noise floor
- Batch Size: **exhausted** — bs=128 is optimal (9% improvement)
- Architecture: **active** — bottleneck_dim promising, in-context injection position exhausted
- Noise Schedule: queued
- Gradient Clipping: queued
- Regularization: queued
