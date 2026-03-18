# Optimization Insights (5s regime)

## Latest Analysis (Batch 91 — 2026-03-18)

**Current best: 0.1156 (depth=1, P_mean=-2.0) | Total improvement: 91.4% from 1.3420**

### Batch 91 Results
```
P_mean=-2.0:              0.1156  << NEW BEST (interaction: shifted from -1.5 at depth=12 to -2.0 at depth=1)
noise_scale=0.01:         0.1184  (slight improvement, ns keeps going lower at depth=1)
noise_scale=0.02:         0.1186
ns=0.05+no_incontext:     0.1187  (combo works but small)
P_mean=-1.0:              0.1355  (much worse)
```

### Conclusion
**P_mean shifts with depth!** At depth=12, P_mean=-1.5 was optimal. At depth=1, P_mean=-2.0 is better. This makes sense: the single-layer model benefits from focusing on higher-noise timesteps even more aggressively — it can only learn coarse structure anyway, so concentrate on the noisiest samples.

### Next Steps — combine ALL improvements
1. P_mean=-2.0 + noise_scale=0.01 + no_incontext (stack all winners)
2. Push P_mean lower (-3.0, -4.0) at depth=1
3. With P_mean=-2.0, re-sweep noise_scale
4. Try P_mean=-2.0 + bs=48 (slightly more steps)

---

## Progression Summary
1.3420 → 0.6815 (LR) → 0.3122 (P_mean) → 0.2021 (noise schedule) → 0.1507 (noise_scale) → 0.1241 (depth=3) → 0.1205 (depth=1) → 0.1192 (ns=0.05) → 0.1156 (P_mean=-2.0)
