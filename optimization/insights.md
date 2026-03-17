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

## What Works (continued)
- **Removing bottleneck** (bn768): consistent improvement, 128→256→384→512→768 is monotonic
- **Late in-context injection** (start10): small additional benefit on top of bottleneck removal
- **Rapid screening validates**: 2-epoch ranking correctly predicted 8-epoch winners

## What Doesn't Work (continued)
- in_context_len=64 + bottleneck combo: diverged (0.1526)
- mlp_ratio=8: much worse (+9.5% at 2ep)
- All dropout configs: hurt performance
- P_std=1.2: much worse. P_mean=-0.4: worse. P_mean=-1.2: marginal.
- Small bottleneck (bn64): very bad (+50% at 2ep)
- in_context_len=8: bad (+26% at 2ep)

## Open Questions
- Is bn768+start10 (0.1132) a real improvement or noise? Need seed runs.
- What about grad clipping with the new architecture?
- Would even later injection (start11) help? Only 1 layer sees tokens then.
- Will results at 128px transfer to 256px?

## Banlist
- warmup=3 + constant LR at blr=1e-3: diverges (NaN)
- cosine schedule: consistently worse than constant in short runs
- bs=256: 30% worse than bs=128
- bn64: much worse
- mlp_ratio=8: worse
- P_std=1.2: much worse
- dropout (any config): hurts

## What Works (further)
- **Grad clipping at 1.0**: small but consistent benefit in rapid screening. At full 8ep, ties with bn768 alone (0.1132).
- **label_drop_prob=0.2**: slight improvement over 0.1 in rapid screening. Marginal at full length.
- Plateau at ~0.1132 across many configs with bn768.

## Category Status
- Optimization/LR: **exhausted** — peak at 2e-3 (bs=128)
- LR Schedule + Warmup: **exhausted** — constant + warmup=1 is best
- Weight Decay: **exhausted** — marginal, within noise floor
- Batch Size: **exhausted** — bs=128 is optimal
- Architecture/Bottleneck: **exhausted** — bn768 is best (remove bottleneck)
- Architecture/In-context: **exhausted** — late injection marginal, removing tokens doesn't help
- Noise Schedule: **exhausted** — all changes hurt
- Dropout/Regularization: **exhausted** — all hurt
- Gradient Clipping: **exhausted** — gc=1.0 marginal, within noise at full length
- Label Drop: **exhausted** — 0.2 marginal
- **Overall: proxy plateau at ~0.1132. 40+ experiments without beating noise floor from 0.1138.**
