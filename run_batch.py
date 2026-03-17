#!/usr/bin/env python3
"""JiT Optimization Daemon — Systematic single-variable search from leaderboard best."""
import json
import subprocess
import re
import os
import time
import random
import csv
from datetime import datetime

QUEUE_FILE = "optimization/queue.json"
LEADERBOARD_FILE = "optimization/leaderboard.md"
HISTORY_FILE = "optimization/all_history.csv"
STATE_FILE = "optimization/search_state.json"
MAX_TIME = 5  # 5s rapid proxy
BATCH_SIZE = 5  # experiments per batch
NOISE_FLOOR = 0.0026  # 1.5 * std from baseline measurements

# Exact config of leaderboard leader b30_crazy_7 (loss=0.2155)
# Verified from results/batch_30/b30_crazy_7/train.log
BEST_CONFIG = {
    "model": "JiT-B/16",
    "img_size": "128",
    "noise_scale": "1.0",
    "bottleneck_dim": "768",
    "shared_adaln": "true",
    "batch_size": "64",
    "blr": "3e-3",
    "epochs": "100",
    "warmup_epochs": "0",
    "max_time": "5",
    "class_num": "10",
    "data_path": "data/imagenette2-320",
    "num_workers": "4",
    "save_last_freq": "100",
    "log_freq": "5",
    "seed": "0",
    "mlp_ratio": "1.0",
    "in_context_len": "16",
    "in_context_start": "10",
    "learned_pos_embed": "true",
    "P_mean": "-2.0",
    "P_std": "0.1",
    "lr_schedule": "constant",
    "weight_decay": "0.0",
    "grad_clip": "0.0",
    "label_drop_prob": "0.1",
}

# Systematic search schedule: ordered by expected impact
# Each entry: (axis_name, [values_to_try], description)
# Values should bracket the current best to find the optimum
SEARCH_SCHEDULE = [
    # 1. Fine-tune LR around current best (3e-3)
    ("blr", ["2e-3", "4e-3", "5e-3", "6e-3", "8e-3"], "LR sweep around current best 3e-3"),
    # 2. Fine-tune P_mean around current best (-2.0)
    ("P_mean", ["-1.0", "-1.5", "-2.5", "-3.0", "-4.0"], "P_mean sweep around current -2.0"),
    # 3. Fine-tune P_std around current best (0.1)
    ("P_std", ["0.05", "0.08", "0.15", "0.2", "0.3"], "P_std sweep around current 0.1"),
    # 4. Batch size (current: 64)
    ("batch_size", ["32", "48", "96", "128", "256"], "Batch size sweep"),
    # 5. In-context length (current: 16)
    ("in_context_len", ["4", "8", "24", "32", "48"], "In-context token count sweep"),
    # 6. In-context start layer (current: 10, depth=12)
    ("in_context_start", ["2", "4", "6", "8", "11"], "In-context injection layer sweep"),
    # 7. MLP ratio (current: 1.0)
    ("mlp_ratio", ["0.5", "0.75", "1.5", "2.0", "3.0"], "MLP expansion ratio sweep"),
    # 8. Gradient clipping (current: 0.0 = disabled)
    ("grad_clip", ["0.5", "1.0", "2.0", "5.0", "10.0"], "Gradient clipping sweep"),
    # 9. Label dropout (current: 0.1)
    ("label_drop_prob", ["0.0", "0.05", "0.15", "0.2", "0.3"], "Label dropout sweep"),
    # 10. Bottleneck dim (current: 768 = no bottleneck)
    ("bottleneck_dim", ["256", "384", "512", "1024", "1536"], "Patch embed bottleneck width sweep"),
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
    log(f"NEW BEST: {exp_id} with {loss:.4f}")

def load_state():
    """Load search state: which axis we're on, current best config."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "schedule_idx": 0,     # which axis in SEARCH_SCHEDULE
        "batch_num": 0,        # global batch counter
        "current_config": BEST_CONFIG.copy(),  # accumulates best values
        "phase": "sweep",      # "sweep" or "refine"
        "refinements_done": 0, # how many refinement rounds for current axis
        "no_improve_count": 0, # consecutive axes with no improvement
    }

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def analyze_batch(queue):
    """Analyze completed batch results. Returns (best_exp, best_loss, all_results)."""
    results = []
    for e in queue:
        if e.get("status") == "done" and isinstance(e.get("result"), (int, float)):
            results.append(e)
    results.sort(key=lambda x: x["result"])
    if not results:
        return None, 999.0, []
    return results[0], results[0]["result"], results

def generate_sweep_batch(state):
    """Generate a batch that sweeps one axis from the search schedule."""
    idx = state["schedule_idx"]
    if idx >= len(SEARCH_SCHEDULE):
        return None  # all axes exhausted

    axis, values, desc = SEARCH_SCHEDULE[idx]
    batch_num = state["batch_num"] + 1
    config = state["current_config"]

    queue = []
    for i, val in enumerate(values):
        changes = {axis: str(val)}
        exp_id = f"s{batch_num}_{axis}_{val}"
        hypothesis = f"Sweep {axis}={val} (current={config.get(axis, 'default')}). {desc}"
        queue.append({
            "exp_id": exp_id,
            "hypothesis": hypothesis,
            "category": "sweep",
            "parent_config": "leaderboard_best",
            "changes": changes,
            "axis": axis,
            "value": str(val),
            "priority": 1,
            "status": "pending",
            "batch": batch_num,
        })

    state["batch_num"] = batch_num
    return queue

def generate_refine_batch(state, best_value, axis, current_best):
    """Generate a refinement batch that zooms in around a winning value."""
    batch_num = state["batch_num"] + 1

    try:
        bv = float(best_value)
        cv = float(current_best)
    except ValueError:
        return None  # can't refine non-numeric

    # Create 5 values bracketing the best, at finer granularity
    delta = abs(bv - cv) / 2 if bv != cv else abs(bv) * 0.2
    if delta == 0:
        delta = 0.1
    values = [bv - 2*delta, bv - delta, bv + delta, bv + 2*delta, bv + 3*delta]

    # Format values appropriately
    if axis in ("batch_size", "in_context_len", "in_context_start", "bottleneck_dim"):
        values = [str(max(1, int(round(v)))) for v in values]
    elif axis == "blr":
        values = [f"{v:.1e}" for v in values if v > 0]
    else:
        values = [f"{v:.4f}" if abs(v) < 1 else f"{v:.2f}" for v in values]

    # Deduplicate and remove the value we already tested
    seen = set()
    unique_values = []
    for v in values:
        if v not in seen and v != str(best_value):
            seen.add(v)
            unique_values.append(v)
    values = unique_values[:BATCH_SIZE]

    if not values:
        return None

    queue = []
    for val in values:
        changes = {axis: str(val)}
        exp_id = f"r{batch_num}_{axis}_{val}"
        hypothesis = f"Refine {axis}={val} around best={best_value}"
        queue.append({
            "exp_id": exp_id,
            "hypothesis": hypothesis,
            "category": "refine",
            "changes": changes,
            "axis": axis,
            "value": str(val),
            "priority": 1,
            "status": "pending",
            "batch": batch_num,
        })

    state["batch_num"] = batch_num
    return queue

def build_args(config, changes, output_dir):
    """Build command-line args from base config + changes."""
    merged = config.copy()
    merged.update(changes)

    args = ["--output_dir", output_dir]
    bool_flags = {"shared_adaln", "learned_pos_embed", "skip_connections",
                  "sandwich_norm", "zero_init_residual_scale"}

    for k, v in merged.items():
        key = f"--{k}"
        if k in bool_flags:
            if v == "true":
                args.append(key)
        else:
            args.extend([key, str(v)])

    return args

def run_experiment(exp, config):
    """Run a single experiment."""
    exp_id = exp["exp_id"]
    batch_num = exp.get("batch", 0)
    output_dir = f"results/batch_{batch_num:02d}/{exp_id}"
    os.makedirs(output_dir, exist_ok=True)

    args = build_args(config, exp.get("changes", {}), output_dir)

    port = 29500 + random.randint(100, 9999)
    cmd = ["torchrun", "--nproc_per_node=1", "--nnodes=1", "--node_rank=0",
           f"--master_port={port}", "main_jit.py"] + args

    log(f"Running: {exp_id} | {exp.get('hypothesis', '')}")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=MAX_TIME + 60)
        output = res.stdout + res.stderr
        with open(f"{output_dir}/train.log", "w") as f:
            f.write(output)

        final_loss = None
        for line in output.split("\n"):
            m = re.search(r'loss: [\d.]+ \(([\d.]+)\)', line)
            if m:
                final_loss = float(m.group(1))
        return final_loss, output
    except Exception as e:
        return None, str(e)

def print_batch_summary(queue, axis, current_val, best_loss_before):
    """Print a clear analysis of batch results."""
    results = [(e["value"], e["result"]) for e in queue
               if e.get("status") == "done" and isinstance(e.get("result"), (int, float))]
    results.sort(key=lambda x: x[1])

    log(f"--- Batch Summary for {axis} (current={current_val}, baseline={best_loss_before:.4f}) ---")
    for val, loss in results:
        delta = loss - best_loss_before
        marker = " << NEW BEST" if delta < -NOISE_FLOOR else ""
        log(f"  {axis}={val}: {loss:.4f} ({delta:+.4f}){marker}")

    if results and results[0][1] < best_loss_before - NOISE_FLOOR:
        log(f"  Winner: {axis}={results[0][0]} (improvement: {best_loss_before - results[0][1]:.4f})")
    else:
        log(f"  No improvement above noise floor ({NOISE_FLOOR})")
    log("---")

def main():
    log("JiT Systematic Search Daemon Starting...")
    log(f"Baseline: b30_crazy_7 | loss: 0.2155")
    log(f"Strategy: sweep one axis at a time, {BATCH_SIZE} experiments per batch")

    state = load_state()

    while True:
        try:
            # Load or create queue
            if not os.path.exists(QUEUE_FILE):
                with open(QUEUE_FILE, "w") as f:
                    json.dump([], f)

            with open(QUEUE_FILE) as f:
                queue = json.load(f)

            pending = [e for e in queue if e["status"] == "pending"]

            # If no pending experiments, analyze last batch and generate next
            if not pending:
                done = [e for e in queue if e.get("status") == "done"]

                if done:
                    # Analyze the completed batch
                    axis = done[0].get("axis", "unknown")
                    current_val = state["current_config"].get(axis, "?")
                    best_loss_before = get_best_loss()

                    best_exp, best_loss, all_results = analyze_batch(queue)
                    print_batch_summary(queue, axis, current_val, best_loss_before)

                    if best_exp and best_loss < best_loss_before - NOISE_FLOOR:
                        # New best found! Update config and leaderboard
                        winning_val = best_exp.get("value", "")
                        state["current_config"][axis] = winning_val
                        state["no_improve_count"] = 0

                        update_leaderboard(
                            best_exp["exp_id"], best_loss,
                            best_exp.get("batch", 0),
                            f"{axis}={winning_val}"
                        )

                        # Check if trend is monotonic - should we continue in same direction?
                        sorted_results = sorted(all_results, key=lambda x: float(x.get("value", 0)) if x.get("value", "").lstrip('-').replace('.','',1).replace('e-','',1).isdigit() else 0)
                        losses = [r["result"] for r in sorted_results]
                        # If best is at the edge, try extending
                        if best_exp == all_results[0]:  # best is lowest or highest tested value
                            if state.get("refinements_done", 0) < 2:
                                log(f"Best at edge of range, refining around {axis}={winning_val}")
                                new_queue = generate_refine_batch(state, winning_val, axis, current_val)
                                if new_queue:
                                    state["refinements_done"] = state.get("refinements_done", 0) + 1
                                    state["phase"] = "refine"
                                    save_state(state)
                                    with open(QUEUE_FILE, "w") as f:
                                        json.dump(new_queue, f, indent=2)
                                    continue
                        # Move to next axis
                        state["schedule_idx"] += 1
                        state["refinements_done"] = 0
                        state["phase"] = "sweep"
                    else:
                        # No improvement, move to next axis
                        state["no_improve_count"] += 1
                        state["schedule_idx"] += 1
                        state["refinements_done"] = 0
                        state["phase"] = "sweep"
                        log(f"No improvement for {axis}. Moving on. ({state['no_improve_count']} consecutive misses)")

                    save_state(state)

                # Generate next batch
                if state["schedule_idx"] >= len(SEARCH_SCHEDULE):
                    log("All axes exhausted! Restarting schedule with updated config.")
                    state["schedule_idx"] = 0
                    state["no_improve_count"] = 0
                    save_state(state)

                new_queue = generate_sweep_batch(state)
                if new_queue is None:
                    log("Nothing to generate. Sleeping...")
                    time.sleep(30)
                    continue

                axis_name = SEARCH_SCHEDULE[state["schedule_idx"]][0]
                log(f"Batch {state['batch_num']}: Sweeping {axis_name}")
                with open(QUEUE_FILE, "w") as f:
                    json.dump(new_queue, f, indent=2)
                save_state(state)
                continue

            # Run pending experiments
            for exp in queue:
                if exp["status"] != "pending":
                    continue
                best_before = get_best_loss()
                exp["status"] = "running"
                with open(QUEUE_FILE, "w") as f:
                    json.dump(queue, f, indent=2)

                loss, output = run_experiment(exp, state["current_config"])

                # Re-read queue (might have changed)
                with open(QUEUE_FILE) as f:
                    queue = json.load(f)

                for e in queue:
                    if e["exp_id"] == exp["exp_id"]:
                        if loss is not None:
                            e["status"] = "done"
                            e["result"] = loss
                            log(f"Result: {exp['exp_id']} -> {loss:.4f}")
                            log_to_history(e["exp_id"], e.get("batch", 0), "done",
                                         f"{loss:.4f}", e.get("hypothesis"), e.get("changes"))
                        else:
                            e["status"] = "failed"
                            e["notes"] = "Error"
                            log(f"Failed: {e['exp_id']}")
                            log_to_history(e["exp_id"], e.get("batch", 0), "failed",
                                         "FAILED", e.get("hypothesis"), e.get("changes"))
                        break

                with open(QUEUE_FILE, "w") as f:
                    json.dump(queue, f, indent=2)

            time.sleep(1)

        except KeyboardInterrupt:
            log("Interrupted. Saving state...")
            save_state(state)
            break
        except Exception as e:
            log(f"Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(10)

if __name__ == "__main__":
    main()
