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

**Batch 5 partial: bottleneck256 promising (-0.0022, just below noise floor). OOM retries needed at bs=128.**

## Batch 6 — Batch Size Validation (128px, 9 epochs)
| Exp ID | Config | Final Loss | Δ vs best (0.1166) | Time | Notes |
|--------|--------|-----------|-------------------|------|-------|
| bs256_ep9 | bs=256, blr=1e-3, 9ep | 0.1518 | +30.2% | 188s | Much worse than bs=128. Confirms bs=128 superiority |

**Batch 6: bs=256 with extra epoch can't match bs=128. More gradient steps > larger batches. Moving to architecture ablations with bs=128 config.**

## Batch 7 — Architecture Ablations (128px, bs=128, 8 epochs, blr=2e-3, constant, warmup=1)
| Exp ID | Change | Final Loss | Δ vs best (0.1166) | Time | Notes |
|--------|--------|-----------|-------------------|------|-------|
| arch_bottleneck256_v2 | bottleneck_dim=256 | 0.1149 | -0.0017 | 308s | Best but below noise floor |
| arch_incontext64 | in_context_len=64 | 0.1164 | -0.0002 | 355s | Marginal |
| arch_mlp6 | mlp_ratio=6.0 | 0.1168 | +0.0002 | 323s | No improvement |
| arch_incontext_start2 | in_context_start=2 | 0.1176 | +0.0010 | 308s | Worse |
| arch_incontext_start0 | in_context_start=0 | 0.1178 | +0.0012 | 319s | Worse |

**Batch 7: no leaderboard improvements (best delta -0.0017 < noise floor 0.0026). bottleneck_dim=256 consistently near-threshold across two configs. Earlier in-context injection hurts. Continue probing bottleneck_dim.**

## Batch 8 — Architecture: Bottleneck Sweep + Combos (128px, bs=128, 8 epochs, blr=2e-3)
| Exp ID | Change | Final Loss | Δ vs best (0.1166) | Time | Notes |
|--------|--------|-----------|-------------------|------|-------|
| arch_bottleneck384 | bottleneck_dim=384 | 0.1141 | -0.0025 | 308s | Near threshold |
| arch_bottleneck512 | bottleneck_dim=512 | 0.1138 | -0.0028 | 308s | **NEW BEST — beats noise floor!** |
| arch_bottleneck256_incontext64 | bottleneck=256+incontext=64 | 0.1526 | +30.9% | 176s | Much worse — combo doesn't work |
| arch_incontext16 | in_context_len=16 | — | — | killed | Interrupted |
| arch_mlp3 | mlp_ratio=3.0 | — | — | skipped | Interrupted |

**Batch 8: bottleneck512 is new best (0.1138). Clear monotonic trend: 128→256→384→512 keeps improving. Combo of bottleneck+incontext diverged.**

## Rapid Exploration — 20 × 2-epoch experiments (128px, bs=128, blr=2e-3)
Top 5 (vs baseline_bn512 0.1755):
| Exp ID | Change | Loss | Δ% |
|--------|--------|------|-----|
| rapid_ic_start10_bn512 | bn512+start10 | 0.1616 | -7.9% |
| rapid_bn768 | bn768 (no bottleneck) | 0.1627 | -7.3% |
| rapid_bn768_mlp3 | bn768+mlp3 | 0.1667 | -5.0% |
| rapid_bn512_ic16_start6 | bn512+ic16+start6 | 0.1682 | -4.2% |
| rapid_ic0_bn512 | bn512+no_incontext | 0.1716 | -2.2% |

Bottom 5 (hurt): pstd_12 (0.3178), bn64 (0.2649), mlp8 (0.2365), drop_both (0.2239), ic8 (0.2212)

## Batch 9 — Validation of Rapid Winners (128px, bs=128, 8 epochs, blr=2e-3)
| Exp ID | Change | Final Loss | Δ vs best (0.1138) | Time | Notes |
|--------|--------|-----------|-------------------|------|-------|
| val_ic_start10_bn768 | bn768+start10 | 0.1132 | -0.0006 | 253s | Best ever but < noise floor |
| val_bn768_mlp3 | bn768+mlp3 | 0.1135 | -0.0003 | 297s | |
| val_bn768 | bn768 | 0.1137 | -0.0001 | 308s | |
| val_bn512_ic16_start6 | bn512+ic16+start6 | 0.1143 | +0.0005 | 260s | |
| val_ic_start10_bn512 | bn512+start10 | 0.1151 | +0.0013 | 267s | |

**Batch 9: val_ic_start10_bn768 (0.1132) is absolute best but delta -0.0006 is within noise. Removing bottleneck (bn768) consistently helps. Late injection (start10) adds small benefit on top.**

## Rapid Exploration Round 2 — 20 × 2-epoch experiments (bn768 base)
Top 5 (vs control bn768+s10 0.1649):
| Exp ID | Change | Loss | Δ% |
|--------|--------|------|-----|
| r2_gradclip1 | gc=1.0+bn768 | 0.1589 | -3.6% |
| r2_labeldrop02 | ld=0.2+bn768 | 0.1613 | -2.2% |
| r2_bn768_s10_gc1 | bn768+s10+gc1 | 0.1618 | -1.9% |
| r2_labeldrop0 | ld=0+bn768 | 0.1621 | -1.7% |
| r2_gradclip2 | gc=2.0+bn768 | 0.1627 | -1.3% |

## Batch 10 — Validation: Grad Clip + Label Drop (128px, bs=128, 8 epochs, blr=2e-3)
| Exp ID | Change | Final Loss | Δ vs best (0.1138) | Time | Notes |
|--------|--------|-----------|-------------------|------|-------|
| val_bn768_gc1 | bn768+gc1.0 | 0.1132 | -0.0006 | 308s | Tied best |
| val_bn768_s10_gc1_ld02 | bn768+s10+gc1+ld0.2 | 0.1132 | -0.0006 | 248s | Tied best |
| val_bn768_gc1_ld02 | bn768+gc1+ld0.2 | 0.1133 | -0.0005 | 568s | |
| val_bn768_labeldrop02 | bn768+ld0.2 | 0.1134 | -0.0004 | 308s | |
| val_bn768_s10_gc1 | bn768+s10+gc1 | 0.1137 | -0.0001 | 256s | |

**Batch 10: All configs cluster at 0.1132-0.1137. No single addition clearly beats noise floor vs bn768 alone. Plateau reached at ~0.1132.**
