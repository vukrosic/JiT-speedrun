#!/usr/bin/env python3
"""JiT Optimization Daemon — Parallel Architecture Branching Strategy."""
import json
import subprocess
import re
import os
import sys
import time
import random
import csv
from datetime import datetime

QUEUE_FILE = "optimization/queue.json"
LEADERBOARD_FILE = "optimization/leaderboard.md"
HISTORY_FILE = "optimization/all_history.csv"
MAX_TIME = 5 # 5s rapid proxy

# Base config (the "Golden Path")
BASE_ARGS = [
    "--model", "JiT-B/16", "--img_size", "128", "--noise_scale", "1.0",
    "--bottleneck_dim", "768", "--shared_adaln", "--batch_size", "64",
    "--blr", "3e-3", "--epochs", "100", "--warmup_epochs", "0",
    "--max_time", "5", "--class_num", "10", "--data_path", "data/imagenette2-320",
    "--num_workers", "4", "--save_last_freq", "100", "--log_freq", "5", "--seed", "0",
]

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def log_to_history(exp_id, batch, status, result, hypothesis, changes):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    file_exists = os.path.isfile(HISTORY_FILE)
    with open(HISTORY_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'batch', 'exp_id', 'status', 'result', 'hypothesis', 'changes'])
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([timestamp, batch, exp_id, status, result, hypothesis, json.dumps(changes)])

def get_best_loss():
    if not os.path.exists(LEADERBOARD_FILE): return 999.0
    with open(LEADERBOARD_FILE) as f:
        m = re.search(r'Active baseline:\*\*\s*\S+\s*\|\s*loss:\s*([\d.]+)', f.read())
        return float(m.group(1)) if m else 999.0

def update_leaderboard(exp_id, loss, batch, change_desc):
    if not os.path.exists(LEADERBOARD_FILE): return
    with open(LEADERBOARD_FILE) as f: lines = f.readlines()
    best_loss = get_best_loss()
    improvement = ((best_loss - loss) / best_loss * 100) if best_loss != 999.0 else 0
    delta = loss - best_loss
    
    new_header = f"Active baseline:** {exp_id} | loss: {loss:.4f} | {datetime.now().strftime('%Y-%m-%d')}\n"
    if len(lines) > 2:
        lines[2] = new_header
    
    rank = 10 + len([l for l in lines if '|' in l and '| Rank |' not in l and '|------|' not in l])
    new_row = f"| {rank} | {exp_id} | {loss:.4f} | {delta:+.4f} | {improvement:.1f}% | {change_desc} | {batch} |\n"
    
    inserted = False
    for i, line in enumerate(lines):
        if '|------|' in line:
            lines.insert(i+1, new_row)
            inserted = True
            break
    if not inserted: lines.append(new_row)
            
    with open(LEADERBOARD_FILE, "w") as f: f.writelines(lines)
    log(f"🏆 LEADERBOARD UPDATED: {exp_id} with {loss:.4f}")

def get_best_config(branch=None):
    """Extract best configuration, optionally for a specific branch."""
    try:
        if not os.path.exists(QUEUE_FILE): return {}, 999.0
        with open(QUEUE_FILE) as f: queue = json.load(f)
        best_exp = None
        min_loss = 999.0
        for e in queue:
            if e.get("status") == "done" and isinstance(e.get("result"), (int, float)):
                if branch and e.get("changes", {}).get("JiT_branch") != branch:
                    continue
                if e["result"] < min_loss:
                    min_loss = e["result"]
                    best_exp = e
        return (best_exp["changes"] if best_exp else {}), min_loss
    except: return {}, 999.0

def generate_next_batch():
    try:
        with open(QUEUE_FILE) as f: queue = json.load(f)
    except: queue = []
    
    if any(e["status"] == "pending" for e in queue): return 
    
    best_global_changes, current_best_loss = get_best_config()
    last_batch_num = max([e.get("batch", 0) for e in queue]) if queue else 0
    next_batch_num = last_batch_num + 1
    
    # Preserve current best as control
    best_overall_exp = None
    for e in queue:
        if e.get("result") == current_best_loss:
            best_overall_exp = e
            break
    
    # We maintain memory of best examples for each branch
    survivors = []
    if best_overall_exp: survivors.append(best_overall_exp)
    for b in ['block_swap', 'conv_bottleneck', 'baseline']:
        b_cfg, b_loss = get_best_config(b)
        if b_loss < 999.0:
            for e in queue:
                if e.get("result") == b_loss and e.get("changes", {}).get("JiT_branch") == b:
                    if e not in survivors: survivors.append(e)
                    break
    queue = survivors
    
    log(f"🧠 ANALYZING BRANCHES. Next Batch: {next_batch_num}. Cooldown (10s)...")
    time.sleep(10)

    axes = {
        "P_std": ["0.1", "0.2", "0.3", "0.5"],
        "P_mean": ["-1.0", "-2.0", "-4.0", "-6.0"],
        "grad_clip": ["0.1", "1.0", "10.0"],
        "label_drop_prob": ["0.0", "0.1", "0.3"],
        "in_context_len": ["16", "32", "64"],
        "bottleneck_dim": ["128", "256", "512", "768"],
        "mlp_ratio": ["1.0", "2.0", "4.0"],
        "shared_adaln": ["true", "false"],
        "skip_connections": ["true", "false"],
    }

    names = {
        "P_std": "Noise_Standard_Deviation",
        "P_mean": "Noise_Mean_Offset",
        "grad_clip": "Gradient_Clipping_Limit",
        "label_drop_prob": "Label_Dropout_Probability",
        "in_context_len": "Context_Token_Length",
        "bottleneck_dim": "Feature_Bottleneck_Width",
        "mlp_ratio": "MLP_Expansion_Factor",
        "shared_adaln": "Adaptive_LayerNorm_Sharing",
        "skip_connections": "UNet_Style_Skip_Connections",
    }

    candidates = []
    
    # 167 Slots for Branch: BlockSwap
    swap_best, _ = get_best_config('block_swap')
    if not swap_best: swap_best = best_global_changes.copy()
    for i in range(167):
        cfg = swap_best.copy()
        cfg["JiT_branch"] = "block_swap"
        axis = random.choice(list(axes.keys()))
        val = random.choice(axes[axis])
        cfg[axis] = val
        name = names.get(axis, axis)
        exp_id = f"B{next_batch_num}_BlockSwap_Mutate_{name}_{val}_{i}"
        hypo = f"[BLOCK-SWAP] Testing if grouped feature shuffling performs better when the {name} is set to {val}."
        candidates.append({"id": exp_id, "changes": cfg, "h": hypo})

    # 167 Slots for Branch: ConvBottleneck
    conv_best, _ = get_best_config('conv_bottleneck')
    if not conv_best: conv_best = best_global_changes.copy()
    for i in range(167):
        cfg = conv_best.copy()
        cfg["JiT_branch"] = "conv_bottleneck"
        axis = random.choice(list(axes.keys()))
        val = random.choice(axes[axis])
        cfg[axis] = val
        name = names.get(axis, axis)
        exp_id = f"B{next_batch_num}_ConvBottle_Mutate_{name}_{val}_{i}"
        hypo = f"[CONV-BOTTLENECK] Evaluating local structure preservation via depthwise conv with {name} optimized to {val}."
        candidates.append({"id": exp_id, "changes": cfg, "h": hypo})

    # 166 Slots for Baseline Refresh
    base_best, _ = get_best_config('baseline')
    if not base_best: base_best = best_global_changes.copy()
    for i in range(166):
        cfg = base_best.copy()
        cfg["JiT_branch"] = "baseline"
        axis = random.choice(list(axes.keys()))
        val = random.choice(axes[axis])
        cfg[axis] = val
        name = names.get(axis, axis)
        exp_id = f"B{next_batch_num}_Baseline_Explore_{name}_{val}_{i}"
        hypo = f"[BASELINE] Refactoring the standard architecture baseline by adjusting {name} to {val}."
        candidates.append({"id": exp_id, "changes": cfg, "h": hypo})

    for i, item in enumerate(candidates):
        queue.append({
            "exp_id": item["id"], "hypothesis": item["h"], "category": item["id"].split("_")[1],
            "changes": item["changes"], "priority": 1, "status": "pending", "batch": next_batch_num
        })
    
    with open(QUEUE_FILE, "w") as f: json.dump(queue, f, indent=2)
    log(f"🚀 GENERATED BATCH {next_batch_num} (Parallel Architecture Branches)")

def run_experiment(exp):
    exp_id = exp["exp_id"]
    batch_num = exp.get("batch", 0)
    output_dir = f"results/batch_{batch_num:02d}/{exp_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    args = BASE_ARGS.copy()
    args.extend(["--output_dir", output_dir])
    
    changes = exp.get("changes", {})
    handled_keys = set()
    for k, v in changes.items():
        key = f"--{k}"
        if v == "true":
            if key not in args: args.append(key)
            handled_keys.add(k)
        elif v == "false":
            while key in args:
                idx = args.index(key)
                if idx + 1 < len(args) and not args[idx+1].startswith("--"): del args[idx:idx+2]
                else: del args[idx]
            handled_keys.add(k)
            
    for k, v in changes.items():
        if k in handled_keys: continue
        key = f"--{k}"
        if key in args:
            idx = args.index(key)
            if idx + 1 < len(args) and not args[idx+1].startswith("--"): args[idx+1] = str(v)
            else: args.insert(idx + 1, str(v))
        else: args.extend([key, str(v)])

    port = 29500 + random.randint(100, 9999)
    cmd = ["torchrun", "--nproc_per_node=1", "--nnodes=1", "--node_rank=0", f"--master_port={port}", "main_jit.py"] + args
    
    log(f"Running: {exp_id} | {exp.get('hypothesis')}")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=MAX_TIME + 60)
        output = res.stdout + res.stderr
        with open(f"{output_dir}/train.log", "w") as f: f.write(output)
        
        final_loss = None
        for line in output.split("\n"):
            m = re.search(r'loss: [\d.]+ \(([\d.]+)\)', line)
            if m: final_loss = float(m.group(1))
        return final_loss, output
    except Exception as e: return None, str(e)

def main():
    log("✨ JiT Architecture Splitting Daemon Starting...")
    while True:
        try:
            if not os.path.exists(QUEUE_FILE):
                with open(QUEUE_FILE, "w") as f: json.dump([], f)
            with open(QUEUE_FILE) as f: queue = json.load(f)
            pending = [e for e in queue if e["status"] == "pending"]
            if not pending:
                generate_next_batch()
                continue
            for exp in queue:
                if exp["status"] != "pending": continue
                best_before = get_best_loss()
                exp["status"] = "running"
                with open(QUEUE_FILE, "w") as f: json.dump(queue, f, indent=2)
                loss, output = run_experiment(exp)
                with open(QUEUE_FILE) as f: queue = json.load(f)
                for e in queue:
                    if e["exp_id"] == exp["exp_id"]:
                        if loss is not None:
                            e["status"] = "done"
                            e["result"] = loss
                            log(f"✅ Result: {loss:.4f}")
                            if loss < best_before:
                                update_leaderboard(e["exp_id"], loss, e.get("batch", 0), e.get("hypothesis", "Architecture Breakthrough"))
                            log_to_history(e["exp_id"], e.get("batch", 0), "done", f"{loss:.4f}", e.get("hypothesis"), e.get("changes"))
                        else:
                            e["status"] = "failed"
                            e["notes"] = "Error"
                            log(f"❌ Failed: {e['exp_id']}")
                            log_to_history(e["exp_id"], e.get("batch", 0), "failed", "FAILED", e.get("hypothesis"), e.get("changes"))
                        break
                with open(QUEUE_FILE, "w") as f: json.dump(queue, f, indent=2)
            time.sleep(2)
        except KeyboardInterrupt: break
        except Exception as e:
            log(f"⚠️ Loop Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
