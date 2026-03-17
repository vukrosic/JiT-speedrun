# Leaderboard

Active baseline:** b30_crazy_7 | loss: 0.2155 | 2026-03-17
**Rule:** Same seed (0) always. 5s training. Any lower loss = better.
**Total improvement:** 76.7% from baseline (1.3420 → 0.3122)
**Full logs:** [all_history.csv](file:///root/workspace/JiT/optimization/all_history.csv)

## Historical Progression

| Rank | Exp ID | Metric | Δ vs Previous | % Improvement | Key Change | Batch |
|------|--------|--------|---------------|---------------|------------|-------|
| 33 | b30_crazy_7 | 0.2155 | -0.0113 | 5.0% | Crazy: in_context_len=16, P_std=0.1, in_context_start=10, shared_adaln=true | 30 |
| 32 | b26_exploit_1 | 0.2268 | -0.0008 | 0.4% | Exploit mutation: P_std=0.3 | 26 |
| 31 | b25_wildcard_4 | 0.2276 | -0.0017 | 0.7% | Wildcard: learned_pos_embed=true, P_std=0.05, mlp_ratio=1.0 | 25 |
| 30 | b24_wildcard_6 | 0.2293 | -0.0008 | 0.3% | Wildcard: learned_pos_embed=true, in_context_start=6 | 24 |
| 29 | b21_wildcard_6 | 0.2301 | -0.0016 | 0.7% | Wildcard: learned_pos_embed=true, in_context_start=4 | 21 |
| 28 | b21_exploit_0 | 0.2317 | -0.0012 | 0.5% | Exploit mutation: in_context_start=2 | 21 |
| 27 | b20_wildcard_4 | 0.2329 | -0.0154 | 6.2% | Wildcard: mlp_ratio=2.0, P_mean=-2.0 | 20 |
| 26 | b20_wildcard_2 | 0.2483 | -0.0128 | 4.9% | Wildcard: mlp_ratio=2.0, sandwich_norm=false | 20 |
| 25 | b19_exploit_1 | 0.2611 | -0.0060 | 2.2% | Exploitation: Mutating ['P_std'] | 19 |
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
