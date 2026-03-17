# Leaderboard

**Active baseline:** control_bs128 | loss: 0.1166 | 2026-03-17
**Noise floor:** std=0.0017 | **Min detectable improvement:** 0.0026 | **Must beat:** 0.1140

## Historical Progression

| Rank | Exp ID | Metric | Δ vs Previous | % Improvement | Key Change | Batch |
|------|--------|--------|---------------|---------------|------------|-------|
| 4 | control_bs128 | 0.1166 | -0.0116 | 9.0% | bs=128, blr=2e-3 (same effective LR, 2x steps) | 5 |
| 3 | lr_1e-3 | 0.1282 | -0.0071 | 5.2% | blr=1e-3 (20x baseline) | 2 |
| 2 | lr_5e-4 | 0.1353 | -0.0570 | 29.6% | blr=5e-4 (10x baseline) | 1 |
| 1 | baseline | 0.1923 | — | — | Default config @ 128px bs=256 | 0 |
