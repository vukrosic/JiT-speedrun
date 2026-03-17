# Optimization Insights

## What Works
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
- Very low LR (2e-5): underfits in 8 epochs

## Surprising Findings
- Default LR (5e-5) was ~20x too low for bs=256 at 128px
- Optimal LR is 1e-3 (20x default). Very sharp cliff above that.
- Gap between 1e-3 and 2e-3 is steep — optimal is likely right at 1e-3

## Open Questions
- Does cosine LR schedule improve over constant at blr=1e-3?
- How sensitive is the model to noise schedule (P_mean, P_std)?
- Does weight decay help or hurt for flow matching?
- Will results at 128px transfer to 256px?

## Category Status
- Optimization/LR: **exhausted** — peak found at 1e-3
- LR Schedule: **active** — batch 3
- Training/Data: queued
- Regularization: queued
- Architecture: queued
