import torch
import time
import os
import sys
import subprocess
from datetime import datetime

# ANSI Color codes
CYAN = "\033[1;36m"
MAGENTA = "\033[1;35m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
RED = "\033[1;31m"
RESET = "\033[0m"
BOLD = "\033[1m"

def get_gpu_stats():
    """Get GPU stats using nvidia-smi."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT
        ).decode("utf-8").strip()
        
        stats = []
        for line in output.split("\n"):
            idx, used, total, util, temp = line.split(", ")
            used, total, util = int(used), int(total), int(util)
            mem_pct = (used / total) * 100
            
            stats.append({
                "id": idx,
                "used": used,
                "total": total,
                "mem_pct": mem_pct,
                "util": util,
                "temp": temp
            })
        return stats
    except Exception:
        return None

def get_cpu_load():
    """Get system load average."""
    try:
        load1, load5, load15 = os.getloadavg()
        return f"{load1:.2f}, {load5:.2f}, {load15:.2f}"
    except Exception:
        return "N/A"

def get_bar(pct, length=10):
    """Create a small progress bar."""
    filled = int(length * pct / 100)
    bar = "█" * filled + "░" * (length - filled)
    
    if pct > 90: color = RED
    elif pct > 70: color = YELLOW
    else: color = GREEN
    
    return f"{color}{bar}{RESET}"

def main():
    os.system('clear')
    print(f"{CYAN}{'='*85}{RESET}")
    print(f" {BOLD}🚀 GPT RESOURCE & PROCESSING MONITOR{RESET}")
    print(f"{CYAN}{'='*85}{RESET}")
    
    print(f" {BOLD}Start Time:{RESET} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" {BOLD}Process ID:{RESET} {os.getpid()}")
    print(f"{CYAN}{'-' * 85}{RESET}")

    # Header
    header = f"{'Time':<10} | {'GPU':<3} | {'Memory %':<15} | {'Usage (MB)':<15} | {'Processing %':<15} | {'Temp':<6}"
    print(f"{BOLD}{header}{RESET}")
    print(f"{CYAN}{'-' * 85}{RESET}")

    try:
        while True:
            current_time = datetime.now().strftime("%H:%M:%S")
            gpu_stats = get_gpu_stats()
            cpu_load = get_cpu_load()

            if gpu_stats:
                for stat in gpu_stats:
                    mem_bar = get_bar(stat['mem_pct'], length=8)
                    proc_bar = get_bar(stat['util'], length=8)
                    
                    # Memory colors
                    mem_color = RED if stat['mem_pct'] > 90 else (YELLOW if stat['mem_pct'] > 70 else GREEN)
                    # Processing colors
                    proc_color = RED if stat['util'] > 90 else (YELLOW if stat['util'] > 70 else GREEN)
                    
                    print(f"{current_time:<10} | "
                          f"#{stat['id']:<2} | "
                          f"{mem_bar} {mem_color}{stat['mem_pct']:>5.1f}%{RESET} | "
                          f"{stat['used']:>5}/{stat['total']:<6} | "
                          f"{proc_bar} {proc_color}{stat['util']:>5}%{RESET} | "
                          f"{stat['temp']}°C")
            else:
                print(f"{current_time:<10} | N/A | {RED}{'No GPU detected':<60}{RESET}")

            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Terminated by user. 👋{RESET}")
    except Exception as e:
        print(f"\n{RED}Error: {e}{RESET}")

if __name__ == "__main__":
    main()