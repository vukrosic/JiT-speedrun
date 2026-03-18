## Just image Transformer (JiT) — Autonomous Optimization

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2511.13720-b31b1b.svg)](https://arxiv.org/abs/2511.13720)

<p align="center">
  <img src="demo/visual.jpg" width="100%">
</p>

This repo is a fork of [JiT](https://arxiv.org/abs/2511.13720) (Back to Basics: Let Denoising Generative Models Denoise) extended with an **autonomous hyperparameter and architecture optimization system**. An AI agent iteratively designs, runs, and analyzes experiments to minimize training loss under a strict 5-second training budget.

### What We Built

Starting from the original JiT-B/16 codebase, we added:

- **Automated experiment runner** (`run_batch.py`) — daemon that executes batches of 5 experiments, parses results, and updates tracking files
- **Live dashboard** (`dashboard.py`) — Flask web UI showing GPU status, experiment progress, leaderboard, and results history
- **Optimization framework** (`optimization/`) — structured search state, experiment queue, leaderboard, insights log, and full history CSV
- **Agent instructions** (`agent.md`) — complete playbook for autonomous ML optimization including strategy, file structure, error handling, and scaling decisions

### Results: 91.4% Loss Reduction

Over **101 batches** (~500 experiments), the agent reduced training loss from **1.3420 to 0.1150** on Imagenette 128px in 5-second runs:

```
1.3420 (baseline)
  -> 0.6815 (LR 1e-3, -49%)           Batch 1: Learning rate was 20x too low
  -> 0.3122 (P_mean=-5.0, -54%)       Batch 9: Shifted timestep sampling to high-noise regime
  -> 0.2021 (noise schedule, -35%)     Batch 25: Fine-tuned P_std, label dropout
  -> 0.1507 (noise_scale=0.1, -25%)   Batch 85: Discovered noise_scale as untapped lever
  -> 0.1205 (depth=1, -20%)           Batch 89: Single transformer layer = max iterations/sec
  -> 0.1150 (mega combo, -4.5%)       Batch 92: Stacked all winning changes
```

### Key Insight: Speed Beats Capacity

In 5-second training (~57 gradient steps), **iteration count matters more than model capacity**. Every winning change made the model faster, not smarter:

| Change | Why It Won |
|---|---|
| `depth=1` | Single transformer layer = maximum iterations per second |
| `noise_scale=0.01` | Easier denoising task = faster convergence per step |
| `P_mean=-2.0, P_std=0.05` | Concentrate on easy timesteps = cleaner gradients |
| `in_context_len=0` | Fewer tokens = faster attention |
| `batch_size=64` | More gradient steps > better gradient estimates |

### Optimal Config

```bash
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 main_jit.py \
  --model JiT-B/16 --img_size 128 --depth 1 \
  --noise_scale 0.01 --P_mean -2.0 --P_std 0.05 \
  --blr 3e-3 --batch_size 64 --epochs 100 --max_time 5 \
  --bottleneck_dim 768 --shared_adaln --learned_pos_embed \
  --mlp_ratio 1.0 --in_context_len 0 --in_context_start 0 \
  --label_drop_prob 0.15 --lr_schedule constant --warmup_epochs 0 \
  --class_num 10 --data_path data/imagenette2-320 --seed 0
```

### What Was Tried and Didn't Work

Over 45+ experiments at the plateau, we tested and rejected:

- **Higher/lower LR** (2e-3, 3.5e-3, 4e-3) — 3e-3 is optimal
- **Larger batch size** (128, 256) — fewer iterations = worse, even with scaled LR
- **Cosine schedule, weight decay, grad clipping** — zero effect
- **Broader timestep sampling** (P_std=0.1, 0.2) — worse
- **JiT-B/32** (patch_size=32) — loses spatial information
- **beta1=0** (no momentum) — much worse
- **Conv front-end** (from F5-TTS paper) — 5x slower per iteration, catastrophic
- **Zero-init residual scaling** (AdaLN-Zero) — hurts at depth=1
- **Smaller bottleneck** (384, 128) — worse
- **Lower noise_scale** (0.005, 0.002, 0.001) — flat below 0.01

### Next Steps

The model is at a genuine plateau. Remaining ideas:

1. **`torch.compile`** — could squeeze 20-40% more iterations per 5s (most promising)
2. **Architecture simplification** — strip RoPE, reduce heads, simplify conditioning
3. **Custom CUDA kernels** — fused attention/MLP for raw speed
4. **Fundamental redesign** — replace transformer block with faster primitive (high risk)

### Project Structure

```
optimization/
  queue.json          # Current batch of experiments
  leaderboard.md      # Best results history (53 entries)
  insights.md         # Analysis and conclusions after each batch
  search_state.json   # Current best config and daemon state
  all_history.csv     # All ~500 experiment results
  recon.md            # Hardware and baseline reconnaissance
  plan.md             # Optimization plan and banlist

run_batch.py          # Automated experiment daemon
dashboard.py          # Flask dashboard (port 5000)
agent.md              # Autonomous agent instructions
main_jit.py           # Training script (added --depth, --beta1, --beta2, --loss_type args)
denoiser.py           # Flow matching denoiser wrapper
model_jit.py          # JiT model architecture (read-only)
engine_jit.py         # Training loop with bf16 autocast
```

### Dashboard

```bash
python3 dashboard.py
```

Starts a Flask server on port 5000 with real-time GPU monitoring, experiment tracking, and leaderboard.

**Remote access via SSH tunnel:**
```bash
ssh -L 5000:localhost:5000 user@remote-server
# Then open http://localhost:5000
```

**Or via ngrok:**
```bash
ngrok http 5000
```

---

### Original Paper

```
@article{li2025jit,
  title={Back to Basics: Let Denoising Generative Models Denoise},
  author={Li, Tianhong and He, Kaiming},
  journal={arXiv preprint arXiv:2511.13720},
  year={2025}
}
```

JiT adopts a minimalist and self-contained design for pixel-level high-resolution image diffusion.
The original implementation was in JAX+TPU. This re-implementation is in PyTorch+GPU.

<p align="center">
  <img src="demo/jit.jpg" width="40%">
</p>

### Original Training

See the original training and evaluation scripts in the [upstream repo](https://github.com/LTH14/JiT).

### Acknowledgements

We thank Google TPU Research Cloud (TRC) for granting us access to TPUs, and the MIT
ORCD Seed Fund Grants for supporting GPU resources.
