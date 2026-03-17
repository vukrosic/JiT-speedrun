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

# Round 3: Go for bigger structural changes — more steps, different batch sizes
# The biggest gain came from 2x gradient steps (bs=128 vs 256 = 9% improvement)
# Try pushing further: bs=64 (4x steps vs bs=256), more epochs
EXPERIMENTS = [
    # Batch size reduction — more gradient steps (the proven strategy)
    {"exp_id": "r3_bs64_bn768", "changes": {"batch_size": "64", "blr": "4e-3", "bottleneck_dim": "768"}, "hypothesis": "bs=64 = 4x steps vs 256. Scale LR linearly."},
    {"exp_id": "r3_bs64_bn768_blr3e3", "changes": {"batch_size": "64", "blr": "3e-3", "bottleneck_dim": "768"}, "hypothesis": "bs=64 with slightly lower LR"},
    {"exp_id": "r3_bs64_bn768_blr2e3", "changes": {"batch_size": "64", "blr": "2e-3", "bottleneck_dim": "768"}, "hypothesis": "bs=64 keep same blr as bs=128"},
    {"exp_id": "r3_bs64_bn768_blr6e3", "changes": {"batch_size": "64", "blr": "6e-3", "bottleneck_dim": "768"}, "hypothesis": "bs=64 push LR even higher"},
    {"exp_id": "r3_bs32_bn768", "changes": {"batch_size": "32", "blr": "8e-3", "bottleneck_dim": "768"}, "hypothesis": "bs=32 = 8x steps. Extreme gradient step increase."},
    {"exp_id": "r3_bs32_bn768_blr4e3", "changes": {"batch_size": "32", "blr": "4e-3", "bottleneck_dim": "768"}, "hypothesis": "bs=32 with moderate LR"},

    # More epochs (stretch time budget)
    {"exp_id": "r3_ep4_bn768", "changes": {"epochs": "4", "bottleneck_dim": "768"}, "hypothesis": "4 epochs at 2-epoch time budget — 2x training"},
    {"exp_id": "r3_ep3_bs64_bn768", "changes": {"epochs": "3", "batch_size": "64", "blr": "4e-3", "bottleneck_dim": "768"}, "hypothesis": "3ep + bs=64 — maximizing steps"},

    # Optimizer params (never tested)
    {"exp_id": "r3_beta2_099_bn768", "changes": {"bottleneck_dim": "768"}, "hypothesis": "Control with bn768 for comparison"},

    # Different in_context_len with bs=64
    {"exp_id": "r3_bs64_ic0_bn768", "changes": {"batch_size": "64", "blr": "4e-3", "bottleneck_dim": "768", "in_context_len": "0"}, "hypothesis": "bs=64 + no in-context — save compute for more steps"},
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
