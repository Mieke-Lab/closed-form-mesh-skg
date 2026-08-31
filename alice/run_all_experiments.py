"""
run_all_experiments.py
----------------------
Sequential runner for exp1 → exp2 → exp3.

For each experiment:
  1. Alice orchestrator (stages 00-03, wait for Bob+Charlie, stages 06-08)
  2. NIST SP 800-22 validation  (10_nist_validation.py)
  3. Paper metrics table        (12_paper_metrics.py)
  4. Copy final keys to channelProb/key/ on Bob and Charlie via SCP

Prints a combined summary table at the end.

Usage:
    python3 run_all_experiments.py [--exps exp1 exp2 exp3] [--wait-timeout 600]
"""
from __future__ import annotations
import os

import argparse
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

BOB_HOST   = os.environ.get("BOB_HOST", "bob@10.0.0.2")
BOB_PASS   = os.environ.get("BOB_PASS", "")
BOB_OUTPUT = "/home/bob/secret-key-generation-collection/magisterSeries/2026/IES_basedPTM_No3/channelProb/output"

CHARLIE_HOST   = os.environ.get("CHARLIE_HOST", "charlie@10.0.0.3")
CHARLIE_PASS   = os.environ.get("CHARLIE_PASS", "")
CHARLIE_OUTPUT = "/home/charlie/secret-key-generation-collection/magisterSeries/2026/IES_basedPTM_No3/channelProb/output"


def _load(script: Path):
    spec = importlib.util.spec_from_file_location(script.stem, script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scp_pull(remote_host: str, password: str, remote_path: str, local_path: Path) -> bool:
    cmd = [
        "sshpass", "-p", password,
        "scp", "-o", "StrictHostKeyChecking=no",
        f"{remote_host}:{remote_path}", str(local_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def pull_peer_keys(exp: str) -> None:
    """Pull Bob and Charlie final keys into local key/ directory."""
    key_dir = BASE_DIR / "key"
    key_dir.mkdir(parents=True, exist_ok=True)
    for host, passwd, remote_out, node in [
        (BOB_HOST, BOB_PASS, BOB_OUTPUT, "bob"),
        (CHARLIE_HOST, CHARLIE_PASS, CHARLIE_OUTPUT, "charlie"),
    ]:
        remote = f"{remote_out}/../key/key_{node}_{exp}.txt"
        local  = key_dir / f"key_{node}_{exp}.txt"
        ok = scp_pull(host, passwd, remote, local)
        status = "OK" if ok else "FAILED"
        print(f"    [SCP] {status}: pulled key_{node}_{exp}.txt")


def run_experiment(exp: str, wait_timeout: int, wait_poll: int) -> dict:
    """Run full pipeline for one experiment. Returns metrics dict."""
    print("\n" + "=" * 70)
    print(f"  EXPERIMENT: {exp}")
    print("=" * 70)

    t_exp_start = time.time()
    success = True

    # Stage: Alice orchestrator (00→01→02→03→wait→06→08)
    try:
        orch = _load(BASE_DIR / "09_alice_orchestrator.py")
        orch.run(exp, wait_timeout=wait_timeout, wait_poll=wait_poll, skip_wait=False)
    except Exception as exc:
        print(f"  [ERROR] Orchestrator failed for {exp}: {exc}")
        success = False

    if not success:
        return {"exp": exp, "status": "ORCH_FAILED"}

    # Stage: NIST validation
    try:
        nist = _load(BASE_DIR / "10_nist_validation.py")
        nist.run(exp)
    except Exception as exc:
        print(f"  [WARN] NIST validation failed for {exp}: {exc}")

    # Pull peer keys into local key/ directory
    pull_peer_keys(exp)

    # Stage: Paper metrics
    metrics_row = {"exp": exp, "status": "OK"}
    try:
        metrics_mod = _load(BASE_DIR / "12_paper_metrics.py")
        metrics_mod.run(exp)
        # Read back the CSV for summary
        import pandas as pd
        csv = OUTPUT_DIR / f"12_paper_metrics_{exp}.csv"
        if csv.is_file():
            row = pd.read_csv(csv).iloc[0].to_dict()
            metrics_row.update(row)
    except Exception as exc:
        print(f"  [WARN] Metrics failed for {exp}: {exc}")

    t_exp_elapsed = time.time() - t_exp_start
    metrics_row["runner_total_s"] = f"{t_exp_elapsed:.1f}"
    return metrics_row


def print_summary(results: list[dict]) -> None:
    cols = ["exp", "N", "KEY_BITS", "BMR_pre", "BMR_post", "BGR", "KGR",
            "ADMM_iters", "NIST", "KDR"]
    print("\n" + "=" * 90)
    print("  COMBINED SUMMARY — all experiments")
    print("=" * 90)
    header = "  " + "  ".join(f"{c:<14}" for c in cols)
    print(header)
    print("  " + "-" * 86)
    for r in results:
        row = "  " + "  ".join(f"{str(r.get(c, 'N/A')):<14}" for c in cols)
        print(row)
    print("=" * 90 + "\n")

    # Also print key file locations
    key_dir = BASE_DIR / "key"
    print("  Key files saved to:")
    for exp in [r.get("exp", "?") for r in results]:
        for node in ("alice", "bob", "charlie"):
            kf = key_dir / f"key_{node}_{exp}.txt"
            status = "OK" if kf.is_file() else "MISSING"
            print(f"    [{status}] {kf.relative_to(BASE_DIR.parent)}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sequential exp1→exp2→exp3 runner")
    parser.add_argument("--exps", nargs="+", default=["exp1", "exp2", "exp3"],
                        choices=["exp1", "exp2", "exp3"],
                        help="Experiments to run (default: exp1 exp2 exp3)")
    parser.add_argument("--wait-timeout", type=int, default=600,
                        help="Seconds to wait for peer error-index files (default 600)")
    parser.add_argument("--wait-poll", type=int, default=10,
                        help="Polling interval in seconds (default 10)")
    args = parser.parse_args(argv)

    t0 = time.time()
    results = []
    for exp in args.exps:
        row = run_experiment(exp, args.wait_timeout, args.wait_poll)
        results.append(row)

    print_summary(results)
    print(f"  Total runner time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
