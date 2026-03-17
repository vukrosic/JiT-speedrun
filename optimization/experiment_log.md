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
