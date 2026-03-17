#!/usr/bin/env python3
"""JiT Optimization Daemon — Continuous autonomous experimentation."""
import json
import subprocess
import re
import os
import time
import random
from datetime import datetime

QUEUE_FILE = "optimization/queue.json"
LEADERBOARD_FILE = "optimization/leaderboard.md"
MAX_TIME = 5 # 5s rapid proxy

# Base config (the "Golden Path")
BASE_ARGS = [
    "--model", "JiT-B/16", "--img_size", "128", "--noise_scale", "1.0",
    "--bottleneck_dim", "768", "--shared_adaln", "--batch_size", "64",
    "--blr", "1.5e-3", "--epochs", "100", "--warmup_epochs", "0",
    "--max_time", "5", "--class_num", "10", "--data_path", "data/imagenette2-320",
    "--num_workers", "4", "--save_last_freq", "100", "--log_freq", "5", "--seed", "0",
]

def get_best_loss():
    if not os.path.exists(LEADERBOARD_FILE): return 999.0
    with open(LEADERBOARD_FILE) as f:
        m = re.search(r'Active baseline:\*\*\s*\S+\s*\|\s*loss:\s*([\d.]+)', f.read())
        return float(m.group(1)) if m else 999.0

def update_leaderboard(exp_id, loss, batch, change_desc):
    with open(LEADERBOARD_FILE) as f: lines = f.readlines()
    best_loss = get_best_loss()
    improvement = ((best_loss - loss) / best_loss * 100) if best_loss != 999.0 else 0
    delta = loss - best_loss
    
    new_header = f"Active baseline:** {exp_id} | loss: {loss:.4f} | {datetime.now().strftime('%Y-%m-%d')}\n"
    lines[2] = new_header
    
    # Add to table (Rank 10+)
    rank = 10 + len([l for l in lines if '|' in l and '| Rank |' not in l and '|------|' not in l])
    new_row = f"| {rank} | {exp_id} | {loss:.4f} | {delta:+.4f} | {improvement:.1f}% | {change_desc} | {batch} |\n"
    
    # Find insertion point (after header)
    for i, line in enumerate(lines):
        if '|------|' in line:
            lines.insert(i+1, new_row)
            break
            
    with open(LEADERBOARD_FILE, "w") as f: f.writelines(lines)
    print(f"🏆 LEADERBOARD UPDATED: {exp_id} with {loss:.4f}")

def get_best_config():
    """Extract best configuration from results."""
    with open(QUEUE_FILE) as f: queue = json.load(f)
    best_exp = None
    min_loss = 999.0
    for e in queue:
        if e.get("status") == "done" and isinstance(e.get("result"), (int, float)):
            if e["result"] < min_loss:
                min_loss = e["result"]
                best_exp = e
    return (best_exp["changes"] if best_exp else {}), min_loss

def generate_next_batch():
    """Analyze results and generate 8 balanced experiments (3 exploit, 3 explore, 2 wildcard)."""
    with open(QUEUE_FILE) as f: queue = json.load(f)
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
    
    print(f"🧹 Clearing queue. Next Batch: {next_batch_num}. Cooldown & Analysis (90s)...")
    time.sleep(90)

    candidates = []
    
    # --- 1. EXPLOITATION (3 slots): Small mutations around best ---
    curr_std = float(best_changes.get("P_std", 0.8))
    curr_mean = float(best_changes.get("P_mean", -0.8))
    curr_blr = float(best_changes.get("blr", 1.5e-3))
    
    exploit_configs = [
        {"P_std": str(round(max(0.05, curr_std - 0.05), 2))},
        {"P_std": str(round(curr_std + 0.05, 2))},
        {"blr": str(curr_blr * 1.2)},
        {"blr": str(curr_blr * 0.8)},
        {"P_mean": str(curr_mean - 0.5)}
    ]
    random.shuffle(exploit_configs)
    for cfg_mod in exploit_configs[:3]:
        cfg = best_changes.copy()
        cfg.update(cfg_mod)
        candidates.append({"id": f"b{next_batch_num}_exploit", "changes": cfg, "h": f"Exploitation: Mutating {list(cfg_mod.keys())}"})

    # --- 2. EXPLORATION (3 slots): Systematic sweep of open axes ---
    explore_configs = [
        {"grad_clip": "1.0"},
        {"label_drop_prob": "0.2"},
        {"in_context_len": "64"},
        {"in_context_start": "2"},
        {"P_std": "0.1"}
    ]
    random.shuffle(explore_configs)
    for cfg_mod in explore_configs[:3]:
        cfg = best_changes.copy()
        cfg.update(cfg_mod)
        candidates.append({"id": f"b{next_batch_num}_explore", "changes": cfg, "h": f"Exploration: Testing {list(cfg_mod.keys())}"})

    # --- 3. WILDCARD (2 slots): Structural architectural shifts ---
    wildcard_configs = [
        {"mlp_ratio": "2.0"},
        {"mlp_ratio": "6.0"},
        {"skip_connections": "true"},
        {"sandwich_norm": "true"},
        {"zero_init_residual_scale": "true"},
        {"learned_pos_embed": "true"},
        {"bottleneck_dim": "512"}
    ]
    random.shuffle(wildcard_configs)
    for cfg_mod in wildcard_configs[:2]:
        cfg = best_changes.copy()
        cfg.update(cfg_mod)
        candidates.append({"id": f"b{next_batch_num}_wildcard", "changes": cfg, "h": f"Wildcard: Random arch change {list(cfg_mod.keys())}"})

    # Final population
    for i, item in enumerate(candidates):
        item["id"] = f"{item['id']}_{i}"
        queue.append({
            "exp_id": item["id"], "hypothesis": item["h"], "category": item["id"].split("_")[1].split("/")[0],
            "changes": item["changes"], "priority": 1, "status": "pending", "batch": next_batch_num
        })
    
    with open(QUEUE_FILE, "w") as f: json.dump(queue, f, indent=2)
    print(f"🚀 GENERATED BATCH {next_batch_num} (8 experiments)")

def run_experiment(exp):
    exp_id = exp["exp_id"]
    output_dir = f"results/batch_{exp['batch']:02d}/{exp_id}"
    os.makedirs(output_dir, exist_ok=True)
    args = BASE_ARGS.copy()
    args.extend(["--output_dir", output_dir])
    changes = exp.get("changes", {})
    for k, v in changes.items():
        key = f"--{k}"
        if v == "true": args.append(key)
        elif key in args: args[args.index(key)+1] = str(v)
        else: args.extend([key, str(v)])
    
    port = 29500 + random.randint(100, 9999)
    cmd = ["torchrun", "--nproc_per_node=1", "--nnodes=1", "--node_rank=0", f"--master_port={port}", "main_jit.py"] + args
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running: {exp_id} | {exp.get('hypothesis')}")
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
    print("✨ JiT Optimization Daemon Starting...")
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
                
                if loss is not None:
                    exp["status"] = "done"
                    exp["result"] = loss
                    print(f"✅ Result: {loss:.4f}")
                    if loss < best_before:
                        update_leaderboard(exp["exp_id"], loss, exp["batch"], exp.get("hypothesis", "Improvement"))
                else:
                    exp["status"] = "failed"
                    exp["notes"] = "OOM" if "CUDA out of memory" in output else "Unknown error"
                    print(f"❌ Failed: {exp['notes']}")
                
                with open(QUEUE_FILE, "w") as f: json.dump(queue, f, indent=2)
                
            time.sleep(2)
        except KeyboardInterrupt: break
        except Exception as e:
            print(f"⚠️ Error in loop: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
