# Experiment Log

## Baseline Runs (128px, bs=256, 8 epochs)
| Exp ID | Seed | Final Loss | Time | Notes |
|--------|------|-----------|------|-------|
| baseline_seed0 | 0 | 0.1904 | 4m13s | 21.5GB VRAM |
| baseline_seed1 | 1 | 0.1937 | 4m13s | 21.9GB VRAM |
| baseline_seed2 | 2 | 0.1928 | 4m14s | 21.4GB VRAM |

**Mean: 0.1923 | Std: 0.0017 | Noise floor (1.5×std): 0.0026 | Must beat: 0.1897**

## Batch 1 — LR Sweep (128px, bs=256, 8 epochs)
| Exp ID | blr | Final Loss | Δ vs Baseline | Time | Notes |
|--------|-----|-----------|---------------|------|-------|
| lr_1e-4 | 1e-4 | 0.1675 | -12.9% | ~4.5m | |
| lr_2e-4 | 2e-4 | 0.1479 | -23.1% | ~4.5m | |
| lr_5e-4 | 5e-4 | 0.1353 | -29.6% | ~4.5m | **Batch 1 best** |

**Batch 1 winner: lr_5e-4 (0.1353). Monotonic improvement — no plateau. Continuing sweep upward.**

## Batch 2 — LR Sweep Continued (128px, bs=256, 8 epochs)
| Exp ID | blr | Final Loss | Δ vs lr_5e-4 | Time | Notes |
|--------|-----|-----------|--------------|------|-------|
| lr_7e-4 | 7e-4 | 0.1322 | -2.3% | 264s | |
| lr_1e-3 | 1e-3 | 0.1282 | -5.2% | 265s | **Batch 2 best** |
| lr_2e-3 | 2e-3 | 0.1517 | +12.1% | 261s | Reversal |
| lr_3e-3 | 3e-3 | 0.1964 | +45.2% | 264s | Near divergence |
| lr_5e-3 | 5e-3 | 0.3277 | +142.2% | 269s | Diverged |

**Batch 2 winner: lr_1e-3 (0.1282). LR peak found. Sharp reversal above 1e-3. Moving to LR schedule.**

## Batch 3 — LR Schedule + Warmup (128px, bs=256, 8 epochs, blr=1e-3)
| Exp ID | Schedule | Warmup | Final Loss | Δ vs lr_1e-3 | Time | Notes |
|--------|----------|--------|-----------|--------------|------|-------|
| sched_cosine | cosine | 1 | 0.1312 | +2.3% | 266s | |
| sched_cosine_warmup2 | cosine | 2 | 0.1306 | +1.9% | 266s | |
| sched_cosine_warmup0 | cosine | 0 | 0.1300 | +1.4% | 264s | |
| warmup0_constant | constant | 0 | 0.1366 | +6.6% | 268s | |
| warmup3_constant | constant | 3 | 0.6064 | diverged | 44s | NaN/diverge — banlist |

**Batch 3: no improvement. Constant LR + warmup=1 remains best. Cosine hurts in short runs. Moving to weight decay.**

## Batch 4 — Weight Decay (128px, bs=256, 8 epochs, blr=1e-3, constant, warmup=1)
| Exp ID | WD | Final Loss | Δ vs best | Time | Notes |
|--------|----|-----------|-----------|------|-------|
| wd_0.01 | 0.01 | 0.1326 | +3.4% | 261s | |
| wd_0.05 | 0.05 | 0.1270 | -0.9% | 262s | |
| wd_0.1 | 0.1 | 0.1297 | +1.2% | 262s | |
| wd_0.3 | 0.3 | 0.1306 | +1.9% | 262s | |
| wd_1e-3 | 0.001 | 0.1271 | -0.9% | 262s | |

**Batch 4: no leaderboard improvement (best delta -0.0012 < noise floor 0.0026). WD has marginal effect. Non-monotonic: 0.01 is worst, 0.05/1e-3 are marginally better. Moving to architecture.**

## Batch 5 — Architecture (128px, bs=256, 8 epochs, blr=1e-3, constant, warmup=1)
| Exp ID | Change | Final Loss | Δ vs best | Time | Notes |
|--------|--------|-----------|-----------|------|-------|
| arch_bottleneck256 | bottleneck_dim=256 | 0.1260 | -0.0022 | 262s | Near threshold |
| arch_no_incontext | in_context_len=0 | 0.1308 | +2.0% | 220s | Conditioning tokens matter |
| arch_mlp6 | mlp_ratio=6.0 | OOM | — | 13s | Retry bs=128 |
| arch_incontext_early | in_context_start=0 | OOM | — | 13s | Retry bs=128 |
| arch_incontext64 | in_context_len=64 | OOM | — | 11s | Retry bs=128 |

**Batch 5 partial: bottleneck256 promising (-0.0022, just below noise floor). OOM retries running at bs=128 with blr=2e-3.**
