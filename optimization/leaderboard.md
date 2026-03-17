# Leaderboard

Active baseline:** batch18_blr3e-3 | loss: 0.2671 | 2026-03-17
**Rule:** Same seed (0) always. 5s training. Any lower loss = better.
**Total improvement:** 76.7% from baseline (1.3420 → 0.3122)

## Historical Progression

| Rank | Exp ID | Metric | Δ vs Previous | % Improvement | Key Change | Batch |
|------|--------|--------|---------------|---------------|------------|-------|
| 24 | batch18_blr3e-3 | 0.2671 | -0.0280 | 9.5% | Pushing LR to 3e-3 with best noise schedule | 18 |
| 23 | batch18_blr2e-3 | 0.2951 | -0.0062 | 2.1% | Pushing LR to 2e-3 with best noise schedule | 18 |
| 22 | r15_pstd02 | 0.3013 | -0.0060 | 2.0% | Ultra-narrow noise focus (P_std=0.2) | 15 |
| 21 | r15_pmean_neg8 | 0.3073 | -0.0018 | 0.6% | Aggressive P_mean (-8.0) | 15 |
| 20 | r15_pmean_neg6 | 0.3091 | -0.0031 | 1.0% | Pushing P_mean even lower (-6.0) | 15 |
| 9 | noise_Pmean_neg5 | 0.3122 | -0.0217 | 6.5% | P_mean=-5.0 — almost all training at near-pure-noise timesteps, learns coarse global structure only | 9 |
| 8 | noise_Pmean_neg2 | 0.3339 | -0.1568 | 31.9% | P_mean=-2.0 — first big shift toward high-noise training focus | 8 |
| 7 | noise_Pmean_neg1 | 0.4907 | -0.0703 | 12.5% | P_mean=-1.0 — initial noise schedule shift toward high-noise timesteps | 7 |
| 6 | bs_64 | 0.5610 | -0.0358 | 6.0% | Batch size 64 — doubles gradient steps per 5s window | 6 |
| 5 | arch_shared_adaln | 0.5968 | -0.0064 | 1.1% | Share conditioning MLP across all blocks — saves 26M params | 5 |
| 4 | arch_bottleneck768 | 0.6032 | -0.0437 | 6.8% | Remove patch embedding bottleneck — full information flow into transformer | 4 |
| 3 | lr_1.5e-3 | 0.6469 | -0.0346 | 5.1% | Base learning rate 1.5e-3 — optimal for fast 5s convergence | 2 |
| 2 | lr_1e-3 | 0.6815 | -0.6605 | 49.2% | Base learning rate 1e-3 — 20x higher than default for rapid learning | 1 |
| 1 | baseline | 1.3420 | — | — | Default config: blr=2e-3, bs=128, bottleneck=128, P_mean=-0.8 | 0 |
