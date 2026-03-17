#!/usr/bin/env python3
"""Run a batch of experiments from a JSON queue file."""
import json
import subprocess
import sys
import re
import os
import time

QUEUE_FILE = "optimization/queue.json"
LOG_FILE = "optimization/experiment_log.md"

# Maximum training time per experiment in seconds (hard rule: 5s rapid iteration)
MAX_TIME = 5

# Base config that all experiments inherit from (matching current best baseline)
BASE_ARGS = [
    "--model", "JiT-B/16",
    "--img_size", "128",
    "--noise_scale", "1.0",
    "--bottleneck_dim", "768",
    "--shared_adaln",
    "--batch_size", "64",
    "--blr", "1.5e-3",
    "--epochs", "100",
    "--warmup_epochs", "0",
    "--max_time", "5",
    "--class_num", "10",
    "--data_path", "data/imagenette2-320",
    "--num_workers", "4",
    "--save_last_freq", "100",
    "--log_freq", "5",
    "--seed", "0",
]

def run_experiment(exp):
    exp_id = exp["exp_id"]
    output_dir = f"results/batch_{exp['batch']:02d}/{exp_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Build args from base + changes
    args = BASE_ARGS.copy()
    args.extend(["--output_dir", output_dir])

    # Apply changes
    changes = exp.get("changes", {})
    if isinstance(changes, dict):
        for k, v in changes.items():
            key = f"--{k}"
            if v == "true":
                # Boolean flag (store_true action)
                args.append(key)
            elif key in args:
                idx = args.index(key)
                args[idx+1] = str(v)
            else:
                args.extend([key, str(v)])

    # Use unique port per experiment to avoid conflicts
    import random
    port = 29500 + random.randint(100, 9999)
    cmd = ["torchrun", "--nproc_per_node=1", "--nnodes=1", "--node_rank=0", f"--master_port={port}", "main_jit.py"] + args

    print(f"\n{'='*60}")
    print(f"Running: {exp_id}")
    print(f"Hypothesis: {exp.get('hypothesis', 'N/A')}")
    print(f"Changes: {changes}")
    print(f"{'='*60}\n")

    start = time.time()
    # Generous subprocess timeout (MAX_TIME + 60s for startup/cleanup)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=MAX_TIME + 60)
    elapsed = time.time() - start

    output = result.stdout + result.stderr

    # Save log
    with open(f"{output_dir}/train.log", "w") as f:
        f.write(output)

    # Extract final loss — take the last reported running average loss
    final_loss = None
    for line in output.split("\n"):
        m = re.search(r'loss: [\d.]+ \(([\d.]+)\)', line)
        if m:
            final_loss = float(m.group(1))

    return final_loss, elapsed, output


def main():
    with open(QUEUE_FILE) as f:
        queue = json.load(f)

    results = []
    for exp in queue:
        if exp["status"] != "pending":
            continue

        exp["status"] = "running"
        # Save queue state
        with open(QUEUE_FILE, "w") as f:
            json.dump(queue, f, indent=2)

        try:
            loss, elapsed, output = run_experiment(exp)
            exp["status"] = "done"
            exp["result"] = loss
            exp["time_s"] = round(elapsed, 1)

            if loss is not None:
                print(f"\n>>> {exp['exp_id']}: loss={loss:.4f} ({elapsed:.0f}s)")
            else:
                print(f"\n>>> {exp['exp_id']}: FAILED to extract loss ({elapsed:.0f}s)")
                exp["status"] = "failed"
                # Check for NaN or OOM
                if "NaN" in output or "nan" in output:
                    exp["status"] = "failed"
                    exp["notes"] = "NaN loss"
                elif "CUDA out of memory" in output:
                    exp["status"] = "failed"
                    exp["notes"] = "OOM"
        except subprocess.TimeoutExpired:
            exp["status"] = "failed"
            exp["notes"] = "timeout"
            print(f"\n>>> {exp['exp_id']}: TIMEOUT")
        except Exception as e:
            exp["status"] = "failed"
            exp["notes"] = str(e)
            print(f"\n>>> {exp['exp_id']}: ERROR {e}")

        results.append(exp)

        # Save queue state after each experiment
        with open(QUEUE_FILE, "w") as f:
            json.dump(queue, f, indent=2)

    # Print summary
    print("\n\n" + "="*70)
    print("BATCH SUMMARY")
    print("="*70)
    done = [e for e in results if e["status"] == "done" and e.get("result")]
    done.sort(key=lambda x: x["result"])
    for e in done:
        marker = " 🏆" if e["result"] < 0.1730 else ""
        print(f"  {e['exp_id']:40s} loss={e['result']:.4f}{marker}")

    failed = [e for e in results if e["status"] == "failed"]
    for e in failed:
        print(f"  {e['exp_id']:40s} FAILED: {e.get('notes', 'unknown')}")


if __name__ == "__main__":
    main()
