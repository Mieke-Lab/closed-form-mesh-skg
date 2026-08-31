"""
09_charlie_orchestrator.py
--------------------------
Run the full Charlie pipeline (stages 04 -> 05 -> 06 -> 07 -> 08) for
a chosen experiment, printing a compact summary at the end.

CLI: python 09_charlie_orchestrator.py --exp exp1
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

STAGES = [
    ("Stage 04 - Preparation",       "04_charlie_preparation.py"),
    ("Stage 05 - ADMM Deobfuscation", "05_charlie_admm_deobfuscation.py"),
    ("Stage 06 - Info Reconciliation","06_charlie_info_reconciliation.py"),
    ("Stage 07 - BMR Evaluation",     "07_charlie_bmr_evaluation.py"),
    ("Stage 08 - Privacy Amplification", "08_charlie_privacy_amplification.py"),
]


def run_stage(label: str, script: str, exp: str) -> tuple[bool, float]:
    path = BASE_DIR / script
    if not path.is_file():
        print(f"[SKIP] {label}: {script} not found.")
        return False, 0.0
    cmd = [sys.executable, str(path), "--exp", exp]
    print(f"\n>>> {label}")
    t0 = time.time()
    proc = subprocess.run(cmd)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"[FAIL] {label} exit={proc.returncode} ({dt:.2f}s)")
        return False, dt
    print(f"[OK] {label} ({dt:.2f}s)")
    return True, dt


def read_text(path: Path) -> str:
    if not path.is_file():
        return "(missing)"
    return path.read_text().strip()


def hash_file(path: Path) -> str:
    if not path.is_file():
        return "(missing)"
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def summarize(exp: str) -> None:
    print("\n==================================================")
    print(f"        Charlie Pipeline Summary ({exp})")
    print("==================================================")
    bmr = OUTPUT_DIR / f"07_charlie_bmr_report_{exp}.txt"
    final = OUTPUT_DIR / f"08_charlie_final_key_{exp}.txt"
    admm = OUTPUT_DIR / f"05_charlie_admm_report_{exp}.txt"
    print(f"\n[ADMM report]\n{read_text(admm)}\n")
    print(f"[BMR report]\n{read_text(bmr)}\n")
    print(f"[Final key]   {read_text(final)}")
    print(f"[Final hash]  {hash_file(final)}")
    print("==================================================\n")


def run(exp: str) -> int:
    print(f"=== Charlie Orchestrator: {exp} ===")
    t0 = time.time()
    elapsed = {}
    for label, script in STAGES:
        ok, dt = run_stage(label, script, exp)
        elapsed[label] = dt
        if not ok and label.startswith("Stage 04"):
            print("[ABORT] preparation failed; halting pipeline.")
            return 1
        if not ok and label.startswith("Stage 05"):
            print("[ABORT] ADMM deobfuscation failed; halting pipeline.")
            return 1
    total = time.time() - t0
    print("\n==================================================")
    print(f"  Total elapsed: {total:.2f}s")
    for label, dt in elapsed.items():
        print(f"    {label:<38} {dt:>7.2f}s")
    print("==================================================")
    summarize(exp)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Charlie pipeline orchestrator")
    parser.add_argument("--exp", required=True, choices=["exp1", "exp2", "exp3"])
    args = parser.parse_args(argv)
    return run(args.exp)


if __name__ == "__main__":
    raise SystemExit(main())
