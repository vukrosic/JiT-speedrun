#!/usr/bin/env python3
"""JiT Optimization Daemon — Continuous autonomous experimentation."""
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
    """Log every experiment to a persistent CSV for meta-analysis."""
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
    
    # Add to table (Rank 10+)
    rank = 10 + len([l for l in lines if '|' in l and '| Rank |' not in l and '|------|' not in l])
    new_row = f"| {rank} | {exp_id} | {loss:.4f} | {delta:+.4f} | {improvement:.1f}% | {change_desc} | {batch} |\n"
    
    # Find insertion point (after header)
    inserted = False
    for i, line in enumerate(lines):
        if '|------|' in line:
            lines.insert(i+1, new_row)
            inserted = True
            break
    if not inserted: lines.append(new_row)
            
    with open(LEADERBOARD_FILE, "w") as f: f.writelines(lines)
    log(f"🏆 LEADERBOARD UPDATED: {exp_id} with {loss:.4f}")

def get_best_config():
    """Extract best configuration from results."""
    try:
        if not os.path.exists(QUEUE_FILE): return {}, 999.0
        with open(QUEUE_FILE) as f: queue = json.load(f)
        best_exp = None
        min_loss = 999.0
        for e in queue:
            if e.get("status") == "done" and isinstance(e.get("result"), (int, float)):
                if e["result"] < min_loss:
                    min_loss = e["result"]
                    best_exp = e
        return (best_exp["changes"] if best_exp else {}), min_loss
    except: return {}, 999.0

def generate_next_batch():
    """Analyze results and generate 8 balanced experiments (Exploit, Crazy, Hyper-Crazy)."""
    try:
        with open(QUEUE_FILE) as f: queue = json.load(f)
    except: queue = []
    
    if any(e["status"] == "pending" for e in queue): return 
    
    best_changes, current_best_loss = get_best_config()
    last_batch_num = max([e.get("batch", 0) for e in queue]) if queue else 0
    next_batch_num = last_batch_num + 1
    
    # Preserve current best context
    best_overall_exp = None
    for e in queue:
        if e.get("result") == current_best_loss:
            best_overall_exp = e
            break
    queue = [best_overall_exp] if best_overall_exp else []
    
    log(f"🧹 Clearing queue. Next Batch: {next_batch_num}. Cooldown & Analysis (10s)...")
    time.sleep(10)

    # --- Evolution Pool ---
    blr = "3e-3" 
    
    axes = {
        "P_std": ["0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.5", "0.7", "1.0", "1.5"],
        "P_mean": ["-0.5", "-1.0", "-2.0", "-3.0", "-4.0", "-5.0", "-6.0", "-8.0", "-10.0", "-12.0"],
        "grad_clip": ["0.01", "0.05", "0.1", "0.5", "1.0", "2.0", "5.0", "10.0", "50.0"],
        "label_drop_prob": ["0.0", "0.05", "0.1", "0.2", "0.3", "0.5", "0.8", "0.9"],
        "in_context_len": ["4", "8", "16", "32", "64", "128", "256", "512"],
        "in_context_start": ["0", "1", "2", "3", "4", "5", "6", "8", "10", "11"],
        "bottleneck_dim": ["16", "32", "64", "128", "256", "384", "512", "768", "1024", "1536"],
        "mlp_ratio": ["0.25", "0.5", "1.0", "2.0", "4.0", "6.0", "8.0", "12.0", "16.0"],
        "weight_decay": ["0.0", "1e-6", "1e-5", "1e-4", "1e-3", "1e-2", "0.1", "0.5", "1.0"],
        "attn_dropout": ["0.0", "0.1", "0.2", "0.4"],
        "proj_dropout": ["0.0", "0.1", "0.2", "0.4"],
        "shared_adaln": ["true", "false"],
        "skip_connections": ["true", "false"],
        "sandwich_norm": ["true", "false"],
        "zero_init_residual_scale": ["true", "false"],
        "learned_pos_embed": ["true", "false"]
    }

    candidates = []
    
    # --- 1. EXPLOITATION (1 slots): Mutation of one best axis ---
    cfg = best_changes.copy()
    cfg["blr"] = blr
    axis = random.choice(list(axes.keys()))
    val = random.choice(axes[axis])
    cfg[axis] = val
    candidates.append({"id": f"b{next_batch_num}_exploit", "changes": cfg, "h": f"Exploit mutation: {axis}={val}"})

    # --- 2. CRAZY WILDCARDS (5 slots): 4-6 random changes ---
    for i in range(5):
        if random.random() > 0.4: cfg = best_changes.copy()
        else: cfg = {}
            
        cfg["blr"] = blr
        num_mutations = random.randint(4, 6)
        mutations = random.sample(list(axes.keys()), min(num_mutations, len(axes)))
        desc = []
        for axis in mutations:
            val = random.choice(axes[axis])
            cfg[axis] = val
            desc.append(f"{axis}={val}")
        candidates.append({"id": f"b{next_batch_num}_crazy", "changes": cfg, "h": f"Crazy: " + ", ".join(desc)})

    # --- 3. HYPER-CRAZY WILDCARDS (2 slots): 8-12 random changes ---
    for i in range(2):
        cfg = {} # Start from scratch
        cfg["blr"] = blr
        num_mutations = random.randint(8, 12)
        mutations = random.sample(list(axes.keys()), min(num_mutations, len(axes)))
        desc = []
        for axis in mutations:
            val = random.choice(axes[axis])
            cfg[axis] = val
            desc.append(f"{axis}={val}")
        candidates.append({"id": f"b{next_batch_num}_hypercrazy", "changes": cfg, "h": f"Hyper-Crazy: " + ", ".join(desc)})

    # Final population
    for i, item in enumerate(candidates):
        item["id"] = f"{item['id']}_{i}"
        queue.append({
            "exp_id": item["id"], "hypothesis": item["h"], "category": item["id"].split("_")[1],
            "changes": item["changes"], "priority": 1, "status": "pending", "batch": next_batch_num
        })
    
    with open(QUEUE_FILE, "w") as f: json.dump(queue, f, indent=2)
    log(f"🚀 GENERATED BATCH {next_batch_num} (8 experiments - HYPER-CRAZY PROTOCOL)")

def run_experiment(exp):
    exp_id = exp["exp_id"]
    batch_num = exp.get("batch", 0)
    output_dir = f"results/batch_{batch_num:02d}/{exp_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    args = BASE_ARGS.copy()
    args.extend(["--output_dir", output_dir])
    
    changes = exp.get("changes", {})
    # Apply changes. Use a set to track which args we've explicitly handled.
    handled_keys = set()
    
    # Process boolean flags first
    for k, v in changes.items():
        key = f"--{k}"
        if v == "true":
            if key not in args: args.append(key)
            handled_keys.add(k)
        elif v == "false":
            while key in args:
                idx = args.index(key)
                if idx + 1 < len(args) and not args[idx+1].startswith("--"):
                    del args[idx:idx+2]
                else:
                    del args[idx]
            handled_keys.add(k)
            
    # Process valued args
    for k, v in changes.items():
        if k in handled_keys: continue
        key = f"--{k}"
        if key in args:
            idx = args.index(key)
            if idx + 1 < len(args) and not args[idx+1].startswith("--"):
                args[idx+1] = str(v)
            else:
                # Flag found, but we want a value. Insert it.
                args.insert(idx + 1, str(v))
        else:
            args.extend([key, str(v)])

    port = 29500 + random.randint(100, 9999)
    cmd = ["torchrun", "--nproc_per_node=1", "--nnodes=1", "--node_rank=0", f"--master_port={port}", "main_jit.py"] + args
    
    log(f"Running: {exp_id} | {exp.get('hypothesis')}")
    log(f"CMD: {' '.join(cmd)}")
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=MAX_TIME + 60)
        output = res.stdout + res.stderr
        with open(f"{output_dir}/train.log", "w") as f: f.write(output)
        
        final_loss = None
        for line in output.split("\n"):
            m = re.search(r'loss: [\d.]+ \(([\d.]+)\)', line)
            if m: final_loss = float(m.group(1))
        
        return final_loss, output
    except Exception as e:
        return None, str(e)

def main():
    log("✨ JiT Optimization Daemon Starting...")
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
                                update_leaderboard(e["exp_id"], loss, e.get("batch", 0), e.get("hypothesis", "Improvement"))
                            
                            log_to_history(e["exp_id"], e.get("batch", 0), "done", f"{loss:.4f}", e.get("hypothesis"), e.get("changes"))
                        else:
                            e["status"] = "failed"
                            e["notes"] = "Error"
                            log(f"❌ Failed: {e['notes']}")
                            log_to_history(e["exp_id"], e.get("batch", 0), "failed", "FAILED", e.get("hypothesis"), e.get("changes"))
                        break
                
                with open(QUEUE_FILE, "w") as f: json.dump(queue, f, indent=2)
                
            time.sleep(2)
        except KeyboardInterrupt: break
        except Exception as e:
            log(f"⚠️ Error in loop: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
