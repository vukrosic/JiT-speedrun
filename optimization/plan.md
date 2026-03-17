# Optimization Plan

## Hardware
- 1x NVIDIA RTX 3090 (24GB VRAM), 96 CPUs, 251GB RAM

## Model
- JiT-B/16: Pixel-space diffusion transformer, ~131M params (varies with mlp_ratio)
- Task: Class-conditional image generation (flow matching, L2 velocity prediction)
- Primary metric: Training loss (lower is better)

## Experiment Config
- img_size: 128, max_time: 5s, seed: 0
- ~5 min per experiment with overhead

## Scaling Decision
- **Hard limit: ≤ 5 min per experiment**
- 128px with bs=64 as current best
- Proxy setup: reduced resolution + reduced data (Imagenette 10-class)

## Active Baseline
- **b30_crazy_7 | loss: 0.2155**
- Config: blr=3e-3, bs=64, bottleneck_dim=768, shared_adaln=true, mlp_ratio=1.0, in_context_len=16, in_context_start=10, learned_pos_embed=true, P_mean=-2.0, P_std=0.1, warmup=0, constant LR, no WD, no grad clip, label_drop=0.1

## Noise Floor
- **Std: 0.0017 | Min detectable improvement: 0.0026**

## Strategy: Systematic Single-Variable Sweep
Each axis is swept with 5 values bracketing the current best. If a sweep finds a winner, check if the trend is at the edge (refine further). Then lock the best value and move to the next axis.

**Sweep order (by expected impact):**
1. blr (current: 3e-3)
2. P_mean (current: -2.0)
3. P_std (current: 0.1)
4. batch_size (current: 64)
5. in_context_len (current: 16)
6. in_context_start (current: 10)
7. mlp_ratio (current: 1.0)
8. grad_clip (current: 0 = off)
9. label_drop_prob (current: 0.1)
10. bottleneck_dim (current: 768)

**Rules:**
- 5 experiments per batch, one variable at a time
- Analyze after each batch before generating next
- If best is at edge of range, refine (up to 2 rounds)
- After all axes done, restart schedule with updated config

## Banlist
- weight_decay: no effect at 5s
- P_std > 0.8: hurts
- skip_connections: overhead, fewer iters
- cosine schedule: worse than constant at 5s
- bs < 32: too noisy
