# Optimization Insights (5s regime)

## What Works
- **Lower P_mean** (noise schedule): Biggest single lever. -0.8→-5.0 took loss from 0.5610 to 0.3122. In 5s, the model can only learn coarse structure — biasing toward high-noise timesteps aligns training with what's achievable.
- **Higher LR**: 1.5e-3 optimal (vs 2e-3 default). Need fast convergence in few steps.
- **No bottleneck (bn768)**: Removing patch embedding compression helps universally.
- **Shared adaLN**: Saves params with no quality loss. Fewer params = faster iters.
- **Smaller batch (64)**: More gradient steps beats larger batches in short training.

## What Doesn't Work
- **Weight decay**: Zero effect at 5s. Too few steps for regularization to matter. BANLIST.
- **P_std > 0.8**: Wider noise distribution hurts. Keep concentrated.
- **Skip connections**: Adds compute overhead, fewer iters in 5s.
- **bs < 32**: Too noisy per step.
- **mlp_ratio changes**: Neither 2x nor 8x helped.

## Exhausted Axes
- LR (peak at 1.5e-3)
- Batch size (64 optimal)
- P_mean (plateau at -2.5 to -5, locked at -5)
- Weight decay (no effect)
- Bottleneck dim (768 optimal)
- Architecture flags (shared_adaln best, others neutral/worse)

## Open
- Grad clipping, label dropout, in-context params, P_std narrower
