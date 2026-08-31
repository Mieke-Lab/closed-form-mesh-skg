"""
17_multirun.py
--------------
Phase C: multi-run statistical analysis of the IES 2026 pipeline.

For each exp in {exp1, exp2, exp3}: invoke `09_alice_orchestrator.py`
N times (default N=10). Snapshot per-run metrics from the produced
artefacts, then aggregate Mean ± SD over the runs.

Captured metrics per run (per exp):
    - KDR (key disagreement rate, %)
    - BMR_pre, BMR_post (bit mismatch at Bob/Charlie vs K_group)
    - KGR (bits/s = 256 / wall_seconds)
    - wall_seconds (end-to-end)
    - Eve_KDR_via_AB, Eve_KDR_via_AC (%)
    - NIST per-test PASS-fraction (8 tests × 1 = 8 booleans)
    - TRSD wall-time stage 05 (ms)

Outputs:
    output/multirun_snapshots/run{R}/{exp}_*.csv|.txt   (per-run artefact)
    output/multirun_summary.csv                          (aggregate stats)

Usage:
    python3 17_multirun.py                  # 10 runs per exp (default)
    python3 17_multirun.py --runs 5         # fewer reps
    python3 17_multirun.py --exp exp1       # one exp only
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
SNAP_DIR = OUTPUT_DIR / "multirun_snapshots"
SNAP_DIR.mkdir(parents=True, exist_ok=True)

ORCHESTRATOR = BASE_DIR / "09_alice_orchestrator.py"

# Artefact filenames per run (from existing pipeline)
ART_PATTERNS = {
    "paper_metrics": "12_paper_metrics_{exp}.csv",
    "nist": "10_nist_results_{exp}.csv",
    "eve": "11_eve_results_{exp}.csv",
    "total_time": "09_total_time_{exp}.txt",
    "admm_report": "05_bob_admm_report_{exp}.txt",
}


def run_orchestrator(exp: str) -> bool:
    """Invoke Alice orchestrator for one full pipeline pass."""
    t0 = time.time()
    proc = subprocess.run(
        ["python3", str(ORCHESTRATOR), "--exp", exp],
        cwd=str(BASE_DIR), capture_output=True, text=True, timeout=420
    )
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"    [ERROR] orchestrator exit {proc.returncode} after {dt:.1f}s")
        print(proc.stderr[-400:])
        return False
    print(f"    [OK] orchestrator finished in {dt:.1f}s")
    return True


def snapshot_run(exp: str, run_idx: int) -> dict:
    """Copy current run's artefacts to snapshots/run{R}/ and extract metrics."""
    rdir = SNAP_DIR / f"run{run_idx:02d}"
    rdir.mkdir(parents=True, exist_ok=True)
    metrics = {"exp": exp, "run": run_idx}

    # Snapshot artefact files
    for key, pat in ART_PATTERNS.items():
        src = OUTPUT_DIR / pat.format(exp=exp)
        if src.is_file():
            shutil.copyfile(src, rdir / src.name)

    # Parse paper_metrics CSV
    pm = OUTPUT_DIR / f"12_paper_metrics_{exp}.csv"
    if pm.is_file():
        df = pd.read_csv(pm)
        if len(df):
            row = df.iloc[0]
            for col in ("KGR", "BGR", "KDR"):
                if col in df.columns:
                    val = row[col]
                    if isinstance(val, str) and val.endswith("%"):
                        val = float(val.rstrip("%"))
                    metrics[col] = float(val)

    # wall_seconds
    tt = OUTPUT_DIR / f"09_total_time_{exp}.txt"
    if tt.is_file():
        for line in tt.read_text().splitlines():
            if "total_wall_seconds" in line and "=" in line:
                metrics["wall_seconds"] = float(line.split("=", 1)[1].strip())
                break

    # TRSD wall-time stage 05
    ar = OUTPUT_DIR / f"05_bob_admm_report_{exp}.txt"
    if ar.is_file():
        for line in ar.read_text().splitlines():
            if line.startswith("wall_seconds="):
                metrics["trsd_wall_seconds"] = float(line.split("=", 1)[1].strip())
                break

    # Eve KDR
    ev = OUTPUT_DIR / f"11_eve_results_{exp}.csv"
    if ev.is_file():
        df = pd.read_csv(ev)
        if len(df):
            row = df.iloc[0]
            for col in df.columns:
                if "kdr_via_AB" in col or col == "eve_kdr_via_AB":
                    metrics["eve_kdr_AB"] = float(str(row[col]).rstrip("%"))
                if "kdr_via_AC" in col or col == "eve_kdr_via_AC":
                    metrics["eve_kdr_AC"] = float(str(row[col]).rstrip("%"))

    # NIST per-test results (8 tests)
    nt = OUTPUT_DIR / f"10_nist_results_{exp}.csv"
    if nt.is_file():
        df = pd.read_csv(nt)
        if "result" in df.columns:
            pass_count = int(np.sum(df["result"] == "PASS"))
            metrics["nist_pass"] = pass_count
            metrics["nist_total"] = len(df)
        if "p_value" in df.columns:
            metrics["nist_p_mean"] = float(df["p_value"].mean())
            metrics["nist_p_min"] = float(df["p_value"].min())

    return metrics


def aggregate(rows: list[dict]) -> pd.DataFrame:
    """Mean ± SD per exp per metric."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    grouped = df.groupby("exp").agg(["mean", "std", "min", "max"])
    grouped.columns = [f"{c[0]}_{c[1]}" for c in grouped.columns]
    return grouped.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--exp", choices=["exp1", "exp2", "exp3"], default=None)
    args = ap.parse_args()

    targets = [args.exp] if args.exp else ["exp1", "exp2", "exp3"]
    all_rows = []

    overall_t0 = time.time()
    for exp in targets:
        print(f"\n=== {exp} — {args.runs} runs ===")
        for run_idx in range(1, args.runs + 1):
            print(f"  [{exp} run {run_idx}/{args.runs}] starting orchestrator...")
            if not run_orchestrator(exp):
                print(f"    [SKIP] run {run_idx}")
                continue
            m = snapshot_run(exp, run_idx)
            all_rows.append(m)
            print(f"    KGR={m.get('KGR','?')}, KDR={m.get('KDR','?')}, "
                  f"wall={m.get('wall_seconds','?')}s, "
                  f"Eve_AB={m.get('eve_kdr_AB','?')}%, "
                  f"NIST={m.get('nist_pass','?')}/{m.get('nist_total','?')}")

    overall = time.time() - overall_t0
    print(f"\n=== All runs complete in {overall:.1f}s ({overall/60:.1f} min) ===\n")

    if not all_rows:
        print("  [ERROR] no successful runs to aggregate")
        return 1

    raw_path = OUTPUT_DIR / "multirun_raw.csv"
    pd.DataFrame(all_rows).to_csv(raw_path, index=False)
    print(f"  Raw runs: {raw_path}")

    summary = aggregate(all_rows)
    summary_path = OUTPUT_DIR / "multirun_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Summary: {summary_path}\n")

    # Print key metrics Mean ± SD
    print("=== Aggregate Mean ± SD ===\n")
    metrics_to_show = [
        ("KGR", "bits/s"), ("BGR", ""), ("wall_seconds", "s"),
        ("eve_kdr_AB", "%"), ("eve_kdr_AC", "%"),
        ("nist_p_mean", ""), ("trsd_wall_seconds", "s"),
    ]
    for _, row in summary.iterrows():
        exp = row["exp"]
        print(f"  {exp}:")
        for col, unit in metrics_to_show:
            mean_key = f"{col}_mean"
            std_key = f"{col}_std"
            if mean_key in row and not pd.isna(row[mean_key]):
                m = row[mean_key]
                s = row[std_key] if (std_key in row and not pd.isna(row[std_key])) else 0.0
                print(f"    {col:<24} = {m:8.4f} ± {s:7.4f} {unit}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
