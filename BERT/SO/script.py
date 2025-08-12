#!/usr/bin/env python
# script.py
"""
Run script.py for many (kappa, T) pairs via subprocess.
Results saved to so_grid_results.csv
"""

import csv, itertools, re, subprocess, sys
from pathlib import Path

# ---------------- user grid ----------------
KAPPAS = [0.01, 0.001, 0.0001, 0.00001]    # 1%, 0.1 %, 0.01 %, 0.001 %
TS     = [5, 10, 20]                    # support-refresh intervals
LOSS_STOP = 1e-3
DEVICE = "cuda:0"                     # or "cpu"
TRAIN_SCRIPT = "main.py"  # adjust path if needed
CSV_OUT = "so_grid_results.csv"
# -------------------------------------------

acc_pat   = re.compile(r":\s*([0-9.]+)%")         # matches "... 84.3%"
step_pat  = re.compile(r"step\s+(\d+)\s+with")    # matches "step 47 with"

def run_one(kappa, T):
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--device", DEVICE,
        "--kappa", str(kappa),
        "--T_sparse", str(T),
        "--loss_stop", str(LOSS_STOP)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out  = proc.stdout + proc.stderr

    # extract metrics
    acc   = float(acc_pat.search(out).group(1))
    steps = int(step_pat.search(out).group(1))
    return steps, acc, out

def main():
    Path(CSV_OUT).write_text("kappa,T,steps,val_acc\n")  # header
    with open(CSV_OUT, "a", newline="") as f:
        wr = csv.writer(f)

        for kappa, T in itertools.product(KAPPAS, TS):
            print(f"\n=== κ={kappa}, T={T} ===")
            try:
                steps, acc, _ = run_one(kappa, T)
                wr.writerow([kappa, T, steps, f"{acc:.1f}"])
                f.flush()
                print(f"→ {steps} steps val acc {acc:.1f}%")
            except Exception as e:
                print(f"Run failed for κ={kappa}, T={T}: {e}")

    print(f"\nAll done. Results saved to {CSV_OUT}")

if __name__ == "__main__":
    main()
