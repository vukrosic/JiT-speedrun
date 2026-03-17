# Optimization Plan

## Hardware
- 1x NVIDIA RTX 3090 (24GB VRAM), 96 CPUs, 251GB RAM

## Model
- JiT-B/16: Pixel-space diffusion transformer, 131M params
- Task: Class-conditional image generation (flow matching, L2 velocity prediction)
- Primary metric: Training loss (lower is better)

## Resolution Decision: Why 256px (not 128px)

The original proxy setup used img_size=128 to keep experiments fast (~45s/epoch). However:
- At 128px, GPU VRAM usage was only ~8GB out of 24GB (34%) — violating the 80% utilization rule.
- At 256px, VRAM usage is ~21GB (87%) — proper utilization.
- 256px produces 256 tokens vs 64 at 128px, which is a fundamentally different attention workload. Results at 128px may not transfer to the real 256px training.
- The tradeoff: experiments take ~21min instead of ~7.5min. But the results are more trustworthy and we're not wasting 66% of the GPU.

**Decision: All experiments run at img_size=256, batch_size=64.**

## Baseline (256px)
- **Seed 0 loss: 0.1471** (10 epochs, ~21min)
- Seeds 1, 2: pending — needed for noise floor measurement
- Config: img_size=256, batch_size=64, blr=5e-5, epochs=10, warmup=1, constant LR, no weight decay, no dropout

## Noise Floor
- Pending: need 3 seed runs to compute std and minimum detectable improvement

## Scaling Decision
- **Tier: reduced_data** — full model at full resolution (256px), small dataset (Imagenette 10-class)
- Each experiment: ~21 minutes (10 epochs × ~2min/epoch)
- Proxy preserves: same architecture, same resolution, same optimizer
- Reduced: dataset size only (10 vs 1000 classes, 9469 vs 1.2M images)

## Experiment Time Budget
- ~21 minutes per experiment
- Batch of 30 experiments ≈ 10.5 hours
- Plan for 2-3 batches before scaling to full ImageNet

## Axes To Explore (Prioritized)

### High Priority — Optimization
1. **Learning rate**: blr in [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4]
2. **LR schedule**: cosine vs constant
3. **Weight decay**: [0.01, 0.05, 0.1]
4. **Warmup epochs**: [0, 1, 2, 3]
5. **Gradient clipping**: [1.0, 0.5, 2.0]
6. **Optimizer betas**: beta2 in [0.95, 0.99, 0.999]

### High Priority — Training
7. **P_mean** (noise schedule center): [-1.2, -0.8, -0.4, 0.0, 0.4]
8. **P_std** (noise schedule spread): [0.4, 0.8, 1.2, 1.6]

### Medium Priority — Regularization
9. **attn_dropout**: [0.05, 0.1]
10. **proj_dropout**: [0.05, 0.1, 0.2]
11. **label_drop_prob**: [0.05, 0.15, 0.2]

### Medium Priority — Architecture
12. **mlp_ratio**: [3.0, 4.0, 6.0]
13. **in_context_len**: [0, 16, 32, 64]
14. **bottleneck_dim**: [64, 128, 256]

### Lower Priority
15. **t_eps**: [1e-2, 5e-2, 1e-1]
16. **noise_scale**: [0.5, 1.0, 1.5]

## Strategy
- Exploration/Exploitation: 60% exploitation / 40% exploration (default)
- Start with LR and noise schedule sweeps (highest expected impact)
- If nothing works after 30 experiments: increase experiment duration to 20 epochs

## Experiment Duration Reasoning
- 10 epochs at 256px: loss curve is still declining (0.1578 → 0.1515 → 0.1471 in epochs 7-9) but relative ranking between configs should be visible
- Will validate by comparing rankings at epoch 7 vs epoch 10 for the first few experiments

## Banlist
(none yet)
