# Optimization Insights (5s regime)

## Latest Analysis (Batches 93-101 — 2026-03-18)

**Current best: 0.1150 (seed=0) | Total improvement: 91.4% from 1.3420 baseline**

### Research-Backed Experiments (Batches 100-101)

Tested architectural ideas from recent flow-matching papers (ManiFlow, F5-TTS):

**Conv front-end (F5-TTS):** ConvNeXt patch embedding with various channel dims:
- bn=768: 0.2251 (only 10 iters, 8.7GB VRAM — way too slow)
- bn=128: 0.1243 (30 iters — better but still fewer than baseline's 57)
- bn=64: 0.1710 (15 iters)
- **Verdict: REJECTED.** Conv processes full-res images before patching, adding enormous overhead. The richer features don't compensate for 2-5x fewer iterations.

**Zero-init residual scaling (AdaLN-Zero/ManiFlow):** 0.1192 (+0.0042 worse)
- **Verdict: REJECTED.** With depth=1, starting at 0.1 scale effectively neuters the single transformer block for many steps.

**Loss parameterization:** x-prediction vs v-prediction vs min-SNR weighting
- Different loss types produce incomparable metrics. With P_std=0.05 (ultra-narrow timestep distribution), loss weighting has minimal effect since all timesteps have similar SNR.

### Deep Plateau Analysis (45+ experiments, batches 93-101)

Every axis has been exhausted:
- **LR:** 3e-3 optimal (2e-3 and 4e-3 both worse)
- **Batch size:** 64 optimal (128 halves iterations → catastrophic)
- **Noise:** ns≤0.01 flat, P_mean=-2.0, P_std=0.05 all optimal
- **Architecture:** shared_adaln=true, learned_pos=true, mlp_ratio=1.0, bn=768 all optimal
- **Optimizer:** beta2=0.85-0.9 gives ~0.0006 marginal gain (below noise threshold)
- **Schedule/regularization:** cosine, weight_decay, grad_clip all make zero difference
- **Model variant:** B/32 much worse (spatial info loss)
- **Conv front-end:** Too slow for 5s regime
- **Zero-init residual:** Hurts at depth=1
- **num_workers:** 4 is optimal

### Why The Plateau Is Real
With depth=1 JiT-B/16 at 128px:
- 57 gradient steps in 5s (0.07s/iter, 0.01s data loading)
- Model already uses bf16 autocast
- Using only 718MB of 24GB VRAM but can't increase batch size (iteration count dominates)
- All hyperparameters, architecture flags, noise schedule, optimizer, LR schedule tested

### Current Best Config
```
depth=1, blr=3e-3, bs=64, noise_scale=0.01, P_mean=-2.0, P_std=0.05,
bottleneck_dim=768, shared_adaln=true, learned_pos_embed=true, mlp_ratio=1.0,
in_context_len=0, in_context_start=0, label_drop=0.15, lr_schedule=constant,
warmup=0, weight_decay=0, grad_clip=0, beta2=0.95, t_eps=0.05
```

### Progression
```
1.3420 (baseline)
→ 0.6815 (LR 1e-3, -49%)
→ 0.3122 (P_mean=-5.0, -54%)
→ 0.2021 (noise schedule tuning, -35%)
→ 0.1507 (noise_scale=0.1, -25%)
→ 0.1205 (depth=1, -20%)
→ 0.1156 (P_mean=-2.0, -4%)
→ 0.1150 (combo, -0.5%)
[HARD PLATEAU — 45+ experiments, 0 significant improvements]
```
