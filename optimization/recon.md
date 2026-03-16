# Reconnaissance

## Hardware
- GPU: 1x NVIDIA RTX 3090 (24GB VRAM)
- CPU: 96 cores
- RAM: 251GB
- Disk: ~25GB free
- No multi-GPU / NVLink

## Repository
- **Model**: JiT (Just image Transformer) — pixel-space diffusion model
- **Architecture**: Vision Transformer with AdaLN modulation, RoPE, SwiGLU FFN, bottleneck patch embedding, in-context class tokens
- **Framework**: PyTorch 2.10.0 + torchrun DDP
- **Default model**: JiT-B/16 (depth=12, hidden=768, heads=12, 131M params)
- **Task**: Class-conditional image generation on ImageNet
- **Training objective**: Flow matching — L2 velocity prediction loss
- **Optimizer**: AdamW, betas=(0.9, 0.95), constant LR with warmup
- **Primary metric**: Training loss (lower is better); FID for generation quality
- **Dataset**: Using Imagenette (10-class ImageNet subset, 9469 train images) since full ImageNet unavailable

## Model Sizes
| Model | Params |
|-------|--------|
| JiT-B/16 | 131.3M |
| JiT-B/32 | 133.2M |
| JiT-L/16 | 459.1M |
| JiT-H/16 | 952.8M |

## Default Hyperparameters
- blr: 5e-5 (scaled by batch_size/256)
- batch_size: 128 per GPU (original 8-GPU setup)
- warmup_epochs: 5
- lr_schedule: constant
- weight_decay: 0.0
- ema_decay1: 0.9999, ema_decay2: 0.9996
- P_mean: -0.8, P_std: 0.8
- noise_scale: 1.0
- label_drop_prob: 0.1 (for CFG)
- attn_dropout: 0.0, proj_dropout: 0.0
- img_size: 256, patch_size: 16

## Proxy Setup
- img_size: 128 (vs 256 original) → 64 tokens vs 256 tokens
- batch_size: 64 (single GPU)
- class_num: 10 (Imagenette)
- ~45 seconds per epoch, 8.2GB VRAM
- 10 epochs per experiment (~7.5 minutes)

## Tunable Axes
### Architecture
- depth, hidden_size, num_heads, mlp_ratio
- bottleneck_dim (patch embed)
- in_context_len, in_context_start
- Attention: qk_norm, qkv_bias

### Optimization
- Learning rate (blr), lr_schedule (constant vs cosine), warmup_epochs
- Optimizer betas, weight_decay
- Gradient clipping (not currently used)

### Regularization
- attn_dropout, proj_dropout
- label_drop_prob

### Training/Data
- P_mean, P_std (noise schedule)
- noise_scale, t_eps
- batch_size
- Precision (currently bf16 autocast)

## Notes
- torch.compile decorators removed due to PyTorch 2.10.0 inductor bug
- Code uses custom attention (not F.scaled_dot_product_attention) with fp32 upcast
