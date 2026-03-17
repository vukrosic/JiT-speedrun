# Optimization Plan

## Hardware
- 1x NVIDIA RTX 3090 (24GB VRAM), 96 CPUs, 251GB RAM

## Model
- JiT-B/16: Pixel-space diffusion transformer, 131M params
- Task: Class-conditional image generation (flow matching, L2 velocity prediction)
- Primary metric: Training loss (lower is better)

## Experiment Config
- img_size: 128, batch_size: 256, epochs: 8
- ~4.5 min per experiment, ~21.9GB VRAM (89%)
- 36 iters/epoch, 288 total iters per experiment

## Scaling Decision
- **Hard limit: ≤ 5 min per experiment**
- 128px with bs=256 to fill VRAM while staying within time budget
- bs=320 OOMs; bs=256 is the maximum
- This is a proxy setup: reduced resolution + reduced data (Imagenette 10-class)
- Proxy preserves: same architecture, same optimizer, same param count

## Baseline
- Pending: need to run 3 seeds at new config (128px, bs=256, 8 epochs)

## Noise Floor
- Pending: need 3 seed runs to compute std and minimum detectable improvement

## Experiment Order
1. **LR sweep** (first batch, ~5 experiments): blr in [1e-5, 2e-5, 5e-5, 1e-4, 2e-4]
2. **LR schedule**: cosine vs constant (with best LR from step 1)
3. **Weight decay**: [0.01, 0.05, 0.1]
4. **Noise schedule**: P_mean, P_std variations
5. **Warmup**: [0, 1, 2, 3] epochs
6. **Gradient clipping**: [0.5, 1.0, 2.0]
7. **Regularization**: attn_dropout, proj_dropout, label_drop_prob
8. **Architecture**: mlp_ratio, in_context_len, bottleneck_dim

Single-variable discipline: each experiment changes at most one thing from its parent config.

## Strategy
- Systematic before exploratory
- Small batches of ~5, analyze after each
- 60/40 exploitation/exploration split
- If no improvement after 15 experiments: reset analysis

## Banlist
(none yet)
