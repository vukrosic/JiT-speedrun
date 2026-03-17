# Optimization Insights

## What Works
(pending batch 1 results)

## What Doesn't Work
(pending)

## Surprising Findings
(pending)

## Open Questions
- Is blr=5e-5 optimal for bs=256 at 128px? (LR scales with batch size)
- Does constant vs cosine LR schedule matter in short runs?
- How sensitive is the model to noise schedule (P_mean, P_std)?
- Does weight decay help or hurt for flow matching?
- Will results at 128px transfer to 256px?

## Category Status
- Optimization: active (starting with LR sweep)
- Training/Data: queued
- Regularization: queued
- Architecture: queued
