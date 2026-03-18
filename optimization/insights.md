# Optimization Insights (5s regime)

## Latest Analysis (Batch 92 — 2026-03-18)

**Current best: 0.1150 (mega combo) | Total improvement: 91.4% from 1.3420**

### Batch 92 Results
```
mega_combo (no_incontext+ns=0.01): 0.1150  << NEW BEST
P_mean=-2.5:                       0.1181  (worse than -2.0)
P_mean=-2.0+ns=0.01:              0.1202  (ns=0.01 alone doesn't help as much)
P_mean=-3.0:                       0.1248  (too aggressive)
P_mean=-4.0:                       0.1535  (way too negative)
```

### Conclusion
**P_mean=-2.0 is optimal at depth=1.** Going more negative (-2.5, -3.0, -4.0) hurts. The combo of P_mean=-2.0 + no_incontext + noise_scale=0.01 gives the best result (0.1150), stacking three small improvements.

### Current Best Config
```
depth=1, blr=3e-3, bs=64, noise_scale=0.05, P_mean=-2.0, P_std=0.05,
bottleneck_dim=768, shared_adaln=true, learned_pos_embed=true, mlp_ratio=1.0,
in_context_len=16, in_context_start=0, label_drop=0.15
```

Mega combo (0.1150): noise_scale=0.01, in_context_len=0

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
```

### What's Left to Try
- Re-sweep LR at the new combo config
- Try eliminating bottleneck (direct patch embed, bn_dim=768 already)
- Try different hidden_size (wider single layer?)
- Try removing learned_pos_embed (save params)
- EMA decay tuning (doesn't affect training loss though)
