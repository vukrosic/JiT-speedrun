#!/usr/bin/env python3
"""Real-time experiment monitoring dashboard."""
import json
import os
import re
import subprocess
import glob
import time
from flask import Flask, jsonify

app = Flask(__name__)

PROJECT = "/root/workspace/JiT"

def get_gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], text=True, timeout=5
        ).strip()
        parts = [x.strip() for x in out.split(",")]
        return {
            "name": parts[0],
            "mem_used_mb": int(parts[1]),
            "mem_total_mb": int(parts[2]),
            "mem_pct": round(int(parts[1]) / int(parts[2]) * 100, 1),
            "gpu_util": int(parts[3]),
            "temp_c": int(parts[4]),
            "power_w": float(parts[5]),
        }
    except Exception as e:
        return {"error": str(e)}

def get_running_experiment():
    try:
        out = subprocess.check_output(["ps", "aux"], text=True, timeout=5)
        for line in out.split("\n"):
            if "main_jit.py" in line and "torchrun" not in line and "grep" not in line:
                m = re.search(r"--output_dir\s+(\S+)", line)
                exp_dir = m.group(1) if m else "unknown"
                exp_name = os.path.basename(exp_dir)
                # Extract key args
                args = {}
                for param in ["batch_size", "blr", "epochs", "bottleneck_dim", "mlp_ratio",
                              "in_context_len", "in_context_start", "grad_clip", "label_drop_prob",
                              "weight_decay", "P_mean", "P_std"]:
                    pm = re.search(rf"--{param}\s+(\S+)", line)
                    if pm:
                        args[param] = pm.group(1)
                return {"running": True, "exp_id": exp_name, "output_dir": exp_dir, "args": args}
        return {"running": False}
    except:
        return {"running": False}

def get_live_training_progress(output_dir):
    log_path = os.path.join(output_dir, "train.log") if output_dir else None
    if not log_path or not os.path.exists(log_path):
        return {"epochs": [], "latest_loss": None}
    epochs = []
    latest_loss = None
    latest_epoch = -1
    latest_iter = -1
    total_iters = -1
    total_epochs = 8
    with open(log_path) as f:
        for line in f:
            m = re.search(r'Start training for (\d+) epochs', line)
            if m:
                total_epochs = int(m.group(1))
            m = re.search(r'Epoch: \[(\d+)\]\s+\[(\d+)/(\d+)\].*loss: ([\d.]+) \(([\d.]+)\)', line)
            if m:
                epoch = int(m.group(1))
                iter_num = int(m.group(2))
                total = int(m.group(3))
                avg_loss = float(m.group(5))
                latest_loss = avg_loss
                latest_epoch = epoch
                latest_iter = iter_num
                total_iters = total
                if re.search(r'eta: 0:00:00', line):
                    epochs.append({"epoch": epoch, "loss": avg_loss})
    return {
        "epochs": epochs, "latest_loss": latest_loss,
        "latest_epoch": latest_epoch, "latest_iter": latest_iter,
        "total_iters": total_iters, "total_epochs": total_epochs,
    }

def get_queue():
    path = os.path.join(PROJECT, "optimization/queue.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return []

def get_leaderboard():
    path = os.path.join(PROJECT, "optimization/leaderboard.md")
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        return f.read()

def get_all_results():
    results = []
    for batch_dir in sorted(glob.glob(os.path.join(PROJECT, "results/batch_*"))):
        batch_name = os.path.basename(batch_dir)
        for exp_dir in sorted(glob.glob(os.path.join(batch_dir, "*"))):
            log = os.path.join(exp_dir, "train.log")
            if os.path.exists(log):
                final_loss = None
                with open(log) as f:
                    for line in f:
                        m = re.search(r'Epoch: \[\d+\].*eta: 0:00:00.*loss: [\d.]+ \(([\d.]+)\)', line)
                        if m:
                            final_loss = float(m.group(1))
                results.append({"batch": batch_name, "exp_id": os.path.basename(exp_dir), "loss": final_loss})
    for exp_dir in sorted(glob.glob(os.path.join(PROJECT, "results/rapid/*"))):
        log = os.path.join(exp_dir, "train.log")
        if os.path.exists(log):
            final_loss = None
            with open(log) as f:
                for line in f:
                    m = re.search(r'Epoch: \[\d+\].*eta: 0:00:00.*loss: [\d.]+ \(([\d.]+)\)', line)
                    if m:
                        final_loss = float(m.group(1))
            results.append({"batch": "rapid", "exp_id": os.path.basename(exp_dir), "loss": final_loss})
    return results

@app.route("/api/status")
def api_status():
    gpu = get_gpu_info()
    running = get_running_experiment()
    progress = get_live_training_progress(running.get("output_dir")) if running.get("running") else {}
    queue = get_queue()
    return jsonify({"gpu": gpu, "running": running, "progress": progress, "queue": queue, "timestamp": time.time()})

@app.route("/api/results")
def api_results():
    return jsonify(get_all_results())

@app.route("/api/leaderboard")
def api_leaderboard():
    return get_leaderboard(), 200, {'Content-Type': 'text/plain'}

@app.route("/")
def index():
    return HTML_PAGE

HTML_PAGE = r"""<!DOCTYPE html>
<html><head>
<title>JiT Optimization Dashboard</title>
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; font-size: 14px; }
  .header { background: linear-gradient(135deg, #161b22 0%, #1c2333 100%); padding: 20px 24px; border-bottom: 2px solid #58a6ff; display: flex; justify-content: space-between; align-items: center; }
  .header h1 { font-size: 20px; color: #f0f6fc; font-weight: 600; }
  .header h1 span { color: #58a6ff; }
  .header .meta { text-align: right; }
  .header .meta .best { color: #3fb950; font-size: 16px; font-weight: bold; }
  .header .meta .target { color: #8b949e; font-size: 12px; }
  .grid { display: grid; grid-template-columns: 300px 1fr; gap: 0; min-height: calc(100vh - 70px); }

  /* Left sidebar */
  .sidebar { background: #161b22; border-right: 1px solid #30363d; padding: 16px; overflow-y: auto; }
  .sidebar-section { margin-bottom: 20px; }
  .sidebar-section h3 { font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; color: #58a6ff; margin-bottom: 10px; font-weight: 600; }
  .gpu-meter { margin-bottom: 8px; }
  .gpu-meter .label-row { display: flex; justify-content: space-between; margin-bottom: 3px; }
  .gpu-meter .label-row .l { color: #8b949e; font-size: 12px; }
  .gpu-meter .label-row .v { color: #f0f6fc; font-size: 12px; font-weight: 600; }
  .bar { height: 6px; background: #21262d; border-radius: 3px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 3px; transition: width 0.5s ease; }
  .bar-fill.mem { background: #bc8cff; }
  .bar-fill.util { background: #58a6ff; }
  .stat-row { display: flex; justify-content: space-between; padding: 4px 0; font-size: 12px; }
  .stat-row .l { color: #8b949e; }
  .stat-row .v { color: #f0f6fc; }

  /* Leaderboard in sidebar */
  .lb-entry { display: flex; align-items: center; gap: 10px; padding: 8px; margin-bottom: 4px; border-radius: 6px; background: #0d1117; }
  .lb-rank { font-size: 16px; font-weight: bold; min-width: 24px; text-align: center; }
  .lb-rank.r1 { color: #d29922; }
  .lb-rank.r2 { color: #8b949e; }
  .lb-rank.r3 { color: #da7b39; }
  .lb-info { flex: 1; min-width: 0; }
  .lb-name { font-weight: 600; color: #f0f6fc; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .lb-loss { color: #3fb950; font-weight: bold; font-size: 15px; }
  .lb-delta { color: #8b949e; font-size: 11px; }

  /* Main content */
  .main { padding: 16px; overflow-y: auto; }

  /* Current batch card */
  .batch-card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; margin-bottom: 16px; overflow: hidden; }
  .batch-header { background: linear-gradient(135deg, #0d2818 0%, #122d1f 100%); padding: 14px 16px; border-bottom: 1px solid #238636; display: flex; justify-content: space-between; align-items: center; }
  .batch-header.idle { background: linear-gradient(135deg, #1c1206 0%, #261a08 100%); border-bottom-color: #d29922; }
  .batch-title { font-size: 16px; font-weight: 600; color: #f0f6fc; }
  .batch-subtitle { color: #8b949e; font-size: 12px; margin-top: 2px; }
  .batch-status { padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
  .batch-status.running { background: #0d2818; color: #3fb950; border: 1px solid #238636; }
  .batch-status.idle { background: #1c1206; color: #d29922; border: 1px solid #d29922; }
  .batch-body { padding: 16px; }

  /* Experiment rows in batch */
  .exp-row { display: grid; grid-template-columns: 32px 1fr 200px 80px 80px; gap: 8px; align-items: center; padding: 10px 8px; border-bottom: 1px solid #21262d; border-radius: 6px; margin-bottom: 2px; }
  .exp-row:last-child { border-bottom: none; }
  .exp-row.is-running { background: #0d2818; border: 1px solid #238636; }
  .exp-row.is-done { opacity: 0.9; }
  .exp-row.is-pending { opacity: 0.5; }
  .exp-row.is-failed { opacity: 0.6; background: #1c0808; }
  .exp-icon { font-size: 18px; text-align: center; }
  .exp-details { min-width: 0; }
  .exp-name { font-weight: 600; color: #f0f6fc; font-size: 14px; }
  .exp-hypothesis { color: #8b949e; font-size: 12px; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .exp-changes { display: flex; gap: 4px; flex-wrap: wrap; margin-top: 4px; }
  .exp-tag { background: #1c2128; border: 1px solid #30363d; padding: 1px 6px; border-radius: 4px; font-size: 11px; color: #bc8cff; font-family: monospace; }
  .exp-progress { }
  .exp-loss { font-size: 18px; font-weight: bold; text-align: right; }
  .exp-loss.good { color: #3fb950; }
  .exp-loss.neutral { color: #c9d1d9; }
  .exp-loss.bad { color: #f85149; }
  .exp-time { color: #8b949e; font-size: 12px; text-align: right; }

  /* Progress bar for running experiment */
  .exp-progress-bar { margin-top: 6px; }
  .progress-text { font-size: 11px; color: #8b949e; margin-bottom: 2px; display: flex; justify-content: space-between; }

  /* Mini epoch chart */
  .mini-chart { display: flex; align-items: flex-end; gap: 1px; height: 40px; margin-top: 6px; }
  .mini-bar { background: #58a6ff; min-width: 8px; flex: 1; border-radius: 2px 2px 0 0; transition: height 0.3s; }
  .mini-bar.best { background: #3fb950; }

  /* Results table */
  .results-card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; overflow: hidden; }
  .results-header { padding: 14px 16px; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between; align-items: center; }
  .results-header h2 { font-size: 14px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
  .results-body { max-height: 400px; overflow-y: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: 8px 10px; color: #8b949e; border-bottom: 1px solid #30363d; font-weight: 500; position: sticky; top: 0; background: #161b22; }
  td { padding: 6px 10px; border-bottom: 1px solid #21262d; }
  tr:hover td { background: #1c2128; }

  .pulse { animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
  .spin { animation: spin 1s linear infinite; display: inline-block; }
  @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
</style>
</head><body>
<div class="header">
  <div>
    <h1><span>JiT</span> Optimization Dashboard</h1>
  </div>
  <div class="meta">
    <div class="best" id="header-best">Best: —</div>
    <div class="target">Must beat: 0.1112 | Noise floor: 0.0026</div>
    <div class="target" id="update-status">Connecting...</div>
  </div>
</div>

<div class="grid">
  <div class="sidebar">
    <!-- GPU -->
    <div class="sidebar-section">
      <h3>GPU</h3>
      <div id="gpu-info">Loading...</div>
    </div>

    <!-- Leaderboard -->
    <div class="sidebar-section">
      <h3>Leaderboard</h3>
      <div id="leaderboard">Loading...</div>
    </div>
  </div>

  <div class="main">
    <!-- Current Batch -->
    <div id="batch-section">Loading...</div>

    <!-- All Results -->
    <div class="results-card" style="margin-top:16px">
      <div class="results-header">
        <h2>All Completed Experiments</h2>
        <span id="results-count" style="color:#8b949e;font-size:12px"></span>
      </div>
      <div class="results-body" id="results-table"></div>
    </div>
  </div>
</div>

<script>
const MUST_BEAT = 0.1112;
let globalBest = 0.1132;

function fmtLoss(v) { return v !== null && v !== undefined ? v.toFixed(4) : '—'; }
function fmtTime(s) { return s ? (s < 60 ? s.toFixed(0)+'s' : Math.floor(s/60)+'m'+Math.round(s%60)+'s') : '—'; }

function lossClass(loss) {
  if (!loss) return 'neutral';
  if (loss <= MUST_BEAT) return 'good';
  if (loss <= globalBest + 0.002) return 'good';
  if (loss <= globalBest + 0.01) return 'neutral';
  return 'bad';
}

function renderGPU(gpu) {
  if (gpu.error) return `<div style="color:#f85149">Error: ${gpu.error}</div>`;
  return `
    <div class="gpu-meter">
      <div class="label-row"><span class="l">VRAM</span><span class="v">${gpu.mem_used_mb}/${gpu.mem_total_mb} MB</span></div>
      <div class="bar"><div class="bar-fill mem" style="width:${gpu.mem_pct}%"></div></div>
    </div>
    <div class="gpu-meter">
      <div class="label-row"><span class="l">Utilization</span><span class="v">${gpu.gpu_util}%</span></div>
      <div class="bar"><div class="bar-fill util" style="width:${gpu.gpu_util}%"></div></div>
    </div>
    <div class="stat-row"><span class="l">Temperature</span><span class="v">${gpu.temp_c}C</span></div>
    <div class="stat-row"><span class="l">Power</span><span class="v">${gpu.power_w}W</span></div>
    <div class="stat-row"><span class="l">${gpu.name}</span></div>
  `;
}

function renderLeaderboard(results) {
  const valid = results.filter(r => r.loss !== null).sort((a,b) => a.loss - b.loss).slice(0, 8);
  if (valid.length > 0) globalBest = valid[0].loss;
  document.getElementById('header-best').textContent = 'Best: ' + (valid.length > 0 ? valid[0].loss.toFixed(4) : '—');

  return valid.map((r, i) => {
    const rc = i === 0 ? 'r1' : i === 1 ? 'r2' : i === 2 ? 'r3' : '';
    const delta = ((r.loss / 0.1923 - 1) * 100).toFixed(1);
    return `
      <div class="lb-entry">
        <div class="lb-rank ${rc}">${i+1}</div>
        <div class="lb-info">
          <div class="lb-name">${r.exp_id}</div>
          <div class="lb-delta">${r.batch} | ${delta}% vs original</div>
        </div>
        <div class="lb-loss">${r.loss.toFixed(4)}</div>
      </div>
    `;
  }).join('');
}

function renderBatch(data) {
  const queue = data.queue || [];
  const running = data.running || {};
  const progress = data.progress || {};

  if (queue.length === 0) {
    return `<div class="batch-card"><div class="batch-header idle"><div><div class="batch-title">No Active Batch</div><div class="batch-subtitle">Queue is empty</div></div><div class="batch-status idle">IDLE</div></div></div>`;
  }

  // Determine batch info
  const batchNum = queue[0].batch || '?';
  const done = queue.filter(e => e.status === 'done').length;
  const total = queue.length;
  const failed = queue.filter(e => e.status === 'failed').length;
  const isRunning = running.running;

  let html = `
    <div class="batch-card">
      <div class="batch-header${isRunning ? '' : ' idle'}">
        <div>
          <div class="batch-title">Batch ${batchNum}</div>
          <div class="batch-subtitle">${done}/${total} completed${failed ? ` | ${failed} failed` : ''}</div>
        </div>
        <div class="batch-status ${isRunning ? 'running' : 'idle'}">${isRunning ? '<span class="spin">&#9881;</span> RUNNING' : 'WAITING'}</div>
      </div>
      <div class="batch-body">
  `;

  for (const exp of queue) {
    const isThis = running.running && running.exp_id === exp.exp_id;
    const statusClass = exp.status === 'done' ? 'is-done' : exp.status === 'running' || isThis ? 'is-running' : exp.status === 'failed' ? 'is-failed' : 'is-pending';
    const icon = exp.status === 'done' ? (exp.result !== null ? '&#10004;' : '&#10060;') : isThis ? '<span class="spin">&#9881;</span>' : exp.status === 'failed' ? '&#10060;' : '&#9711;';

    // Changes tags
    const changes = exp.changes || {};
    const tags = Object.entries(changes).map(([k,v]) => `<span class="exp-tag">${k}=${v}</span>`).join('');

    // Loss display
    let lossHtml = '';
    let timeHtml = '';
    if (exp.status === 'done' && typeof exp.result === 'number') {
      const lc = lossClass(exp.result);
      const delta = ((exp.result - globalBest) * 1000).toFixed(1);
      const sign = exp.result <= globalBest ? '' : '+';
      lossHtml = `<div class="exp-loss ${lc}">${exp.result.toFixed(4)}</div><div style="font-size:11px;color:#8b949e;text-align:right">${sign}${delta} vs best</div>`;
      timeHtml = `<div class="exp-time">${fmtTime(exp.time_s)}</div>`;
    } else if (isThis && progress.latest_loss) {
      const lc = lossClass(progress.latest_loss);
      lossHtml = `<div class="exp-loss ${lc} pulse">${progress.latest_loss.toFixed(4)}</div>`;

      // Progress info
      const ep = progress.latest_epoch >= 0 ? progress.latest_epoch + 1 : 0;
      const totalEp = progress.total_epochs || 8;
      const pct = progress.total_iters > 0 ? Math.round(((progress.latest_epoch * progress.total_iters + progress.latest_iter) / (totalEp * progress.total_iters)) * 100) : 0;
      timeHtml = `
        <div class="exp-progress-bar">
          <div class="progress-text"><span>Epoch ${ep}/${totalEp}</span><span>${pct}%</span></div>
          <div class="bar"><div class="bar-fill util" style="width:${pct}%"></div></div>
        </div>
      `;

      // Mini epoch chart
      if (progress.epochs && progress.epochs.length > 0) {
        const maxL = Math.max(...progress.epochs.map(e => e.loss));
        const minL = Math.min(...progress.epochs.map(e => e.loss));
        const range = maxL - minL * 0.9 || 1;
        const bars = progress.epochs.map(e => {
          const h = Math.max(4, ((e.loss - minL * 0.9) / range) * 36);
          const bc = e.loss <= globalBest ? 'best' : '';
          return `<div class="mini-bar ${bc}" style="height:${h}px" title="E${e.epoch}: ${e.loss.toFixed(4)}"></div>`;
        }).join('');
        timeHtml += `<div class="mini-chart">${bars}</div>`;
      }
    } else if (exp.status === 'failed') {
      lossHtml = `<div class="exp-loss bad">FAILED</div>`;
      timeHtml = exp.notes ? `<div class="exp-time">${exp.notes}</div>` : '';
    }

    html += `
      <div class="exp-row ${statusClass}">
        <div class="exp-icon">${icon}</div>
        <div class="exp-details">
          <div class="exp-name">${exp.exp_id}</div>
          <div class="exp-hypothesis">${exp.hypothesis || ''}</div>
          <div class="exp-changes">${tags}</div>
        </div>
        <div class="exp-progress">${timeHtml}</div>
        <div>${lossHtml}</div>
        <div></div>
      </div>
    `;
  }

  html += `</div></div>`;
  return html;
}

function updateDashboard() {
  fetch('/api/status')
    .then(r => r.json())
    .then(data => {
      document.getElementById('update-status').textContent = 'Updated: ' + new Date().toLocaleTimeString();
      document.getElementById('gpu-info').innerHTML = renderGPU(data.gpu);
      document.getElementById('batch-section').innerHTML = renderBatch(data);
    })
    .catch(e => { document.getElementById('update-status').textContent = 'Error: ' + e.message; });
}

function updateResults() {
  fetch('/api/results')
    .then(r => r.json())
    .then(results => {
      const valid = results.filter(r => r.loss !== null).sort((a,b) => a.loss - b.loss);
      document.getElementById('leaderboard').innerHTML = renderLeaderboard(results);
      document.getElementById('results-count').textContent = valid.length + ' experiments';

      const rows = valid.map((r, i) => {
        const lc = r.loss <= globalBest + 0.001 ? 'color:#3fb950;font-weight:bold' : r.loss < 0.15 ? '' : 'color:#f85149';
        const delta = ((r.loss - globalBest) / globalBest * 100).toFixed(1);
        const ds = parseFloat(delta) <= 0 ? `<span style="color:#3fb950">${delta}%</span>` : `<span style="color:#8b949e">+${delta}%</span>`;
        return `<tr><td style="color:#58a6ff">${i+1}</td><td style="font-weight:500">${r.exp_id}</td><td style="${lc}">${r.loss.toFixed(4)}</td><td>${ds}</td><td style="color:#8b949e">${r.batch}</td></tr>`;
      }).join('');
      document.getElementById('results-table').innerHTML = `<table><tr><th>#</th><th>Experiment</th><th>Loss</th><th>vs Best</th><th>Source</th></tr>${rows}</table>`;
    });
}

updateDashboard();
updateResults();
setInterval(updateDashboard, 3000);
setInterval(updateResults, 10000);
</script>
</body></html>
"""

if __name__ == "__main__":
    port = int(os.environ.get("DASH_PORT", 5000))
    print("=" * 50)
    print("  JiT Optimization Dashboard")
    print(f"  http://0.0.0.0:{port}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=port, debug=False)
