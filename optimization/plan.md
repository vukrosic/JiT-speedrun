# Optimization Plan

## Hardware
- 1x NVIDIA RTX 3090 (24GB VRAM), 96 CPUs, 251GB RAM

## Model
- JiT-B/16: Pixel-space diffusion transformer, 131M params
- Task: Class-conditional image generation
- Primary metric: Training loss (L2 velocity prediction, lower is better)

## Baseline
- **Mean loss: 0.1772** (seeds 0/1/2: 0.1793, 0.1741, 0.1783)
- Config: img_size=128, batch_size=64, blr=5e-5, epochs=10, warmup=1, constant LR, no weight decay, no dropout

## Noise Floor
- Standard deviation: 0.0028
- **Minimum detectable improvement: 0.0042** (1.5 × 0.0028)
- Baseline threshold: loss must be < 0.1730 to count as improvement

## Scaling Decision
- **Tier: reduced_data** — full model, reduced resolution (128 vs 256) and small dataset (Imagenette)
- Each experiment: ~7.5 minutes (10 epochs × 45s)
- Proxy preserves: same model architecture, same depth/width ratio, same optimizer
- Reduced: image resolution (128 vs 256 → 4x fewer tokens), dataset (10 vs 1000 classes)

## Experiment Time Budget
- ~7.5 minutes per experiment
- Batch of 30 experiments ≈ 3.75 hours
- Plan for 2-3 batches before scaling up

## Axes To Explore (Prioritized)

### High Priority — Optimization
1. **Learning rate**: blr in [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4]
2. **LR schedule**: cosine vs constant
3. **Weight decay**: [0.01, 0.05, 0.1]
4. **Warmup epochs**: [0, 1, 2, 3]
5. **Gradient clipping**: [1.0, 0.5, 2.0] (not currently used)
6. **Optimizer betas**: beta2 in [0.95, 0.99, 0.999]

### High Priority — Training
7. **P_mean** (noise schedule center): [-1.2, -0.8, -0.4, 0.0, 0.4]
8. **P_std** (noise schedule spread): [0.4, 0.8, 1.2, 1.6]
9. **Batch size**: [32, 64, 128] (with LR scaling)

### Medium Priority — Regularization
10. **attn_dropout**: [0.05, 0.1]
11. **proj_dropout**: [0.05, 0.1, 0.2]
12. **label_drop_prob**: [0.05, 0.15, 0.2]
13. **Weight decay + dropout combos**

### Medium Priority — Architecture
14. **mlp_ratio**: [3.0, 4.0, 6.0]
15. **in_context_len**: [0, 16, 32, 64]
16. **bottleneck_dim**: [64, 128, 256]

### Lower Priority
17. **t_eps**: [1e-2, 5e-2, 1e-1]
18. **noise_scale**: [0.5, 1.0, 1.5]
19. **Activation function changes**

## Strategy
- Exploration/Exploitation: 60% exploitation / 40% exploration (default)
- Start with LR and noise schedule sweeps (highest expected impact)
- If nothing works after 30 experiments: increase experiment duration to 20 epochs

## Experiment Duration Reasoning
- 10 epochs: loss curve is still dropping but has differentiated between configs
- Will validate by running first 5 experiments at 20 epochs and comparing rankings

## Banlist
(none yet)
