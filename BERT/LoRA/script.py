#!/usr/bin/env python
# script.py
"""
Launch lora_rte32_lossstop_bar.py for multiple LoRA ranks via subprocess.
Captures step-count, final loss, and validation accuracy → lora_grid_results.csv
"""

import csv, subprocess, sys, re, itertools
from pathlib import Path

# ---------------- user grid ----------------
RANKS      = [2, 4, 8, 16]          # LoRA r values to sweep
LOSS_STOP  = 1e-3                   # same convergence threshold as SO
DEVICE     = "cuda:0"                 # or "cpu"
TRAIN_FILE = "main.py"     # path to LoRA training script
CSV_FILE   = "lora_grid_results.csv"
# -------------------------------------------

acc_re   = re.compile(r":\s*([0-9.]+)%")        # "... 84.3%"
step_re  = re.compile(r"step\s+(\d+)\s+with")   # "step 47 with"

def run_rank(rank):
    cmd = [
        sys.executable, TRAIN_FILE,
        "--device", DEVICE,
        "--rank",   str(rank),
        "--loss_stop", str(LOSS_STOP)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out  = proc.stdout + proc.stderr
    
    # parse stdout
    acc   = float(acc_re.search(out).group(1))
    steps = int(step_re.search(out).group(1))
    return steps, acc, out

def main():
    Path(CSV_FILE).write_text("rank,steps,val_acc\n")  # header
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        for r in RANKS:
            print(f"\n=== LoRA rank r={r} ===")
            try:
                steps, acc, _ = run_rank(r)
                writer.writerow([r, steps, f"{acc:.1f}"])
                f.flush()
                print(f"→ {steps} steps val acc {acc:.1f}%")
            except Exception as e:
                print(f"Run failed for r={r}: {e}")

    print(f"\nDone. Results in {CSV_FILE}")

if __name__ == "__main__":
    main()
