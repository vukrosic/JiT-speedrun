#!/usr/bin/env python3
"""Rapid diverse architecture exploration — 20 short experiments (~1 min each)."""
import json
import subprocess
import re
import os
import time
import random

QUEUE_FILE = "optimization/rapid_queue.json"

# Current best config as base (control_bs128 + bottleneck512)
BASE_ARGS = [
    "--model", "JiT-B/16",
    "--img_size", "128",
    "--noise_scale", "1.0",
    "--batch_size", "128",
    "--blr", "2e-3",
    "--epochs", "2",
    "--warmup_epochs", "0",
    "--class_num", "10",
    "--data_path", "data/imagenette2-320",
    "--num_workers", "4",
    "--save_last_freq", "100",
    "--log_freq", "10",
    "--seed", "0",
]

# 20 diverse architecture experiments
EXPERIMENTS = [
    # Baseline control at 2 epochs
    {"exp_id": "rapid_baseline", "changes": {}, "hypothesis": "2-epoch baseline with default arch"},
    {"exp_id": "rapid_baseline_bn512", "changes": {"bottleneck_dim": "512"}, "hypothesis": "2-epoch baseline with current best (bn512)"},

    # Bottleneck extremes
    {"exp_id": "rapid_bn768", "changes": {"bottleneck_dim": "768"}, "hypothesis": "bottleneck=hidden_size, essentially removing bottleneck"},
    {"exp_id": "rapid_bn64", "changes": {"bottleneck_dim": "64"}, "hypothesis": "Very narrow bottleneck — aggressive compression"},

    # MLP ratio variations
    {"exp_id": "rapid_mlp2", "changes": {"mlp_ratio": "2.0"}, "hypothesis": "Much smaller FFN"},
    {"exp_id": "rapid_mlp8", "changes": {"mlp_ratio": "8.0"}, "hypothesis": "Very large FFN — 2x default"},
    {"exp_id": "rapid_mlp5_bn512", "changes": {"mlp_ratio": "5.0", "bottleneck_dim": "512"}, "hypothesis": "Bigger FFN + bigger bottleneck combo"},

    # In-context variations
    {"exp_id": "rapid_ic128", "changes": {"in_context_len": "128"}, "hypothesis": "4x in-context tokens"},
    {"exp_id": "rapid_ic8", "changes": {"in_context_len": "8"}, "hypothesis": "Very few in-context tokens"},
    {"exp_id": "rapid_ic0_bn512", "changes": {"in_context_len": "0", "bottleneck_dim": "512"}, "hypothesis": "No in-context + big bottleneck"},

    # In-context start variations with bottleneck
    {"exp_id": "rapid_ic_start6_bn512", "changes": {"in_context_start": "6", "bottleneck_dim": "512"}, "hypothesis": "Late injection (layer 6/12) + big bottleneck"},
    {"exp_id": "rapid_ic_start10_bn512", "changes": {"in_context_start": "10", "bottleneck_dim": "512"}, "hypothesis": "Very late injection (layer 10/12) + big bottleneck"},

    # Dropout experiments
    {"exp_id": "rapid_attn_drop01", "changes": {"attn_dropout": "0.1"}, "hypothesis": "Light attention dropout for regularization"},
    {"exp_id": "rapid_proj_drop01", "changes": {"proj_dropout": "0.1"}, "hypothesis": "Light projection dropout"},
    {"exp_id": "rapid_drop_both", "changes": {"attn_dropout": "0.05", "proj_dropout": "0.05"}, "hypothesis": "Light dropout everywhere"},

    # Noise schedule
    {"exp_id": "rapid_pmean_neg12", "changes": {"P_mean": "-1.2"}, "hypothesis": "Shift noise distribution toward lower noise levels"},
    {"exp_id": "rapid_pmean_neg04", "changes": {"P_mean": "-0.4"}, "hypothesis": "Shift noise toward higher noise levels"},
    {"exp_id": "rapid_pstd_12", "changes": {"P_std": "1.2"}, "hypothesis": "Wider noise distribution"},

    # Combos — wild cards
    {"exp_id": "rapid_bn768_mlp3", "changes": {"bottleneck_dim": "768", "mlp_ratio": "3.0"}, "hypothesis": "No bottleneck + smaller FFN — reallocate params"},
    {"exp_id": "rapid_bn512_ic16_start6", "changes": {"bottleneck_dim": "512", "in_context_len": "16", "in_context_start": "6"}, "hypothesis": "Big bottleneck + fewer late-injected tokens"},
]

def run_experiment(exp):
    exp_id = exp["exp_id"]
    output_dir = f"results/rapid/{exp_id}"
    os.makedirs(output_dir, exist_ok=True)

    args = BASE_ARGS.copy()
    args.extend(["--output_dir", output_dir])

    changes = exp.get("changes", {})
    if isinstance(changes, dict):
        for k, v in changes.items():
            key = f"--{k}"
            if key in args:
                idx = args.index(key)
                args[idx+1] = str(v)
            else:
                args.extend([key, str(v)])

    port = 29500 + random.randint(100, 9999)
    cmd = ["torchrun", "--nproc_per_node=1", "--nnodes=1", "--node_rank=0", f"--master_port={port}", "main_jit.py"] + args

    print(f"\n{'='*60}")
    print(f"Running: {exp_id}")
    print(f"Changes: {changes}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    elapsed = time.time() - start

    output = result.stdout + result.stderr

    with open(f"{output_dir}/train.log", "w") as f:
        f.write(output)

    # Extract final loss from last epoch's last iteration
    final_loss = None
    for line in output.split("\n"):
        m = re.search(r'Epoch: \[\d+\].*\]\s+eta: 0:00:00.*loss: [\d.]+ \(([\d.]+)\)', line)
        if m:
            final_loss = float(m.group(1))

    return final_loss, elapsed, output

def main():
    results = []
    for exp in EXPERIMENTS:
        try:
            loss, elapsed, output = run_experiment(exp)
            exp["result"] = loss
            exp["time_s"] = round(elapsed, 1)
            if loss is not None:
                print(f">>> {exp['exp_id']}: loss={loss:.4f} ({elapsed:.0f}s)")
            else:
                if "CUDA out of memory" in output:
                    exp["result"] = "OOM"
                    print(f">>> {exp['exp_id']}: OOM ({elapsed:.0f}s)")
                else:
                    exp["result"] = "FAILED"
                    print(f">>> {exp['exp_id']}: FAILED ({elapsed:.0f}s)")
        except subprocess.TimeoutExpired:
            exp["result"] = "TIMEOUT"
            print(f">>> {exp['exp_id']}: TIMEOUT")
        except Exception as e:
            exp["result"] = str(e)
            print(f">>> {exp['exp_id']}: ERROR {e}")
        results.append(exp)

    # Save results
    with open("optimization/rapid_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print sorted summary
    print("\n\n" + "="*70)
    print("RAPID EXPLORATION SUMMARY (sorted by loss)")
    print("="*70)
    done = [e for e in results if isinstance(e.get("result"), float)]
    done.sort(key=lambda x: x["result"])
    for e in done:
        print(f"  {e['exp_id']:40s} loss={e['result']:.4f}  {e.get('hypothesis','')[:50]}")

    failed = [e for e in results if not isinstance(e.get("result"), float)]
    if failed:
        print("\nFailed:")
        for e in failed:
            print(f"  {e['exp_id']:40s} {e.get('result', 'unknown')}")

if __name__ == "__main__":
    main()
