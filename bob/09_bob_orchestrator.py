"""
09_bob_orchestrator.py
----------------------
Run the Bob-side KeyCome Group-Key SKG pipeline end-to-end:
    04 -> 05 -> 06 -> 07 -> 08

Prints a concise summary at the end with BMR pre/post, recovered key hash,
and the wall-clock time per stage.
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
    "04_bob_preparation.py",
    "05_bob_admm_deobfuscation.py",
    "06_bob_info_reconciliation.py",
    "07_bob_bmr_evaluation.py",
    "08_bob_privacy_amplification.py",
]


def run_stage(stage: str, exp: str) -> tuple[bool, float]:
    cmd = [sys.executable, str(BASE_DIR / stage), "--exp", exp]
    print(f"\n>>> {stage}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd)
    return proc.returncode == 0, time.perf_counter() - t0


def hash_file(path: Path) -> str:
    if not path.is_file():
        return "<missing>"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def parse_bmr_report(exp: str) -> dict[str, str]:
    rep = OUTPUT_DIR / f"07_bob_bmr_report_{exp}.txt"
    out = {}
    if rep.is_file():
        for line in rep.read_text().splitlines():
            if "=" in line:
                key, val = line.split("=", 1)
                out[key.strip()] = val.strip()
    return out


def run(exp: str) -> int:
    print(f"=== Bob KeyCome SKG pipeline: {exp} ===")
    timings: dict[str, float] = {}
    for stage in STAGES:
        ok, elapsed = run_stage(stage, exp)
        timings[stage] = elapsed
        if not ok:
            print(f"[FATAL] {stage} failed after {elapsed:.2f}s")
            return 1

    print("\n=== Summary ===")
    bmr = parse_bmr_report(exp)
    for key in ("BMR_pre_reconciliation",
                "BMR_post_reconciliation",
                "BMR_post_BCH"):
        if key in bmr:
            print(f"  {key:30s} {bmr[key]}")

    rec_hash = hash_file(OUTPUT_DIR / f"05_bob_recovered_key_{exp}.csv")
    rcn_hash = hash_file(OUTPUT_DIR / f"06_bob_reconciled_key_{exp}.csv")
    fin_path = OUTPUT_DIR / f"08_bob_final_key_{exp}.txt"
    fin_key = fin_path.read_text().strip() if fin_path.is_file() else "<missing>"
    print(f"  recovered_key_sha256_16   {rec_hash}")
    print(f"  reconciled_key_sha256_16  {rcn_hash}")
    print(f"  final_key_hex             {fin_key}")

    print("\n=== Stage timings (s) ===")
    for stage, t in timings.items():
        print(f"  {stage:38s} {t:7.2f}")
    total = sum(timings.values())
    print(f"  {'TOTAL':38s} {total:7.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bob orchestrator")
    parser.add_argument("--exp", required=True, choices=["exp1", "exp2", "exp3"])
    args = parser.parse_args(argv)
    return run(args.exp)


if __name__ == "__main__":
    raise SystemExit(main())
