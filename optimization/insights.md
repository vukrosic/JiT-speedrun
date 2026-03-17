# Optimization Insights

## What Works
- **Higher learning rates**: Monotonic improvement from 2e-5 → 5e-4. No plateau yet.
  - 2e-5: 0.2314 (worse than baseline 5e-5)
  - 5e-5: 0.1923 (baseline)
  - 1e-4: 0.1675 (-12.9%)
  - 2e-4: 0.1479 (-23.1%)
  - 5e-4: 0.1353 (-29.6%)
- The trend is strongly monotonic — next batch should push higher (7e-4, 1e-3, 2e-3, 3e-3, 5e-3)

## What Doesn't Work
- Very low LR (2e-5): underfits in 8 epochs, worse than baseline

## Surprising Findings
- Default LR (5e-5) was ~10x too low for bs=256 at 128px
- Even 10x the default LR (5e-4) shows no sign of divergence

## Open Questions
- Where does LR peak? Need to find the reversal point.
- Does constant vs cosine LR schedule matter in short runs?
- How sensitive is the model to noise schedule (P_mean, P_std)?
- Does weight decay help or hurt for flow matching?
- Will results at 128px transfer to 256px?

## Category Status
- Optimization/LR: **active** — continuing LR sweep upward (batch 2)
- Training/Data: queued
- Regularization: queued
- Architecture: queued
