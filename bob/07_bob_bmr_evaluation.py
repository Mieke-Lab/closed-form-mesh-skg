"""
07_bob_bmr_evaluation.py
------------------------
Stage 07 of the KeyCome Group-Key SKG pipeline (Bob side).

Computes Bit Mismatch Rate (BMR) of Bob's keys against Alice's ground-truth
K_group at three checkpoints:

  pre_reconciliation : 05_bob_recovered_key vs Alice K_group
  post_reconciliation: 06_bob_reconciled_key vs Alice K_group
  post_BCH_only      : same as pre_reconciliation here, kept explicit for the
                       paper's metric table (Bob already runs BCH inside ADMM).

Alice's ground-truth K_group is fetched via SSH from Alice's output directory.

Output:
    channelProb/output/07_bob_bmr_report_{exp}.txt
"""
from __future__ import annotations
import os

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

ALICE_HOST = os.environ.get("ALICE_HOST", "alice@10.0.0.1")
ALICE_PASS = os.environ.get("ALICE_PASS", "")
ALICE_OUTPUT = (
    "/home/alice/secret-key-generation-collection/magisterSeries/2026/"
    "IES_basedPTM_No3/channelProb/output"
)


def fetch_alice_k_group(exp: str) -> Path:
    remote = f"{ALICE_OUTPUT}/03_alice_K_group_{exp}.csv"
    local = OUTPUT_DIR / f"07_alice_K_group_{exp}.csv"
    cmd = [
        "sshpass", "-p", ALICE_PASS,
        "scp", "-o", "StrictHostKeyChecking=no",
        f"{ALICE_HOST}:{remote}", str(local),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return local
    except subprocess.CalledProcessError as exc:
        msg = exc.stderr.decode(errors="ignore") if exc.stderr else ""
        raise FileNotFoundError(
            f"Could not fetch Alice K_group via SCP: {msg.strip()}"
        )


def load_key(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    return np.concatenate([
        df["real"].to_numpy(dtype=int),
        df["imag"].to_numpy(dtype=int),
    ])


def bmr(a: np.ndarray, b: np.ndarray) -> tuple[float, int, int]:
    n = min(a.size, b.size)
    if n == 0:
        return 1.0, 0, 0
    diff = int(np.sum(a[:n] != b[:n]))
    return diff / n, diff, n


def run(exp: str) -> None:
    print(f"--- Stage 07: Bob BMR Evaluation ({exp}) ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rec_path = OUTPUT_DIR / f"05_bob_recovered_key_{exp}.csv"
    rcn_path = OUTPUT_DIR / f"06_bob_reconciled_key_{exp}.csv"
    if not rec_path.is_file():
        raise FileNotFoundError(f"Stage 05 key missing: {rec_path}")
    if not rcn_path.is_file():
        raise FileNotFoundError(f"Stage 06 key missing: {rcn_path}")

    print("    [SCP] fetching Alice K_group ground truth ...")
    alice_path = fetch_alice_k_group(exp)
    alice_key = load_key(alice_path)
    bob_pre = load_key(rec_path)
    bob_post = load_key(rcn_path)

    pre_rate, pre_diff, pre_n = bmr(bob_pre, alice_key)
    post_rate, post_diff, post_n = bmr(bob_post, alice_key)
    bch_rate, bch_diff, bch_n = pre_rate, pre_diff, pre_n  # Bob's BCH already inside ADMM stage

    rep_path = OUTPUT_DIR / f"07_bob_bmr_report_{exp}.txt"
    with rep_path.open("w") as fh:
        fh.write(f"exp={exp}\n")
        fh.write(f"key_length_bits={alice_key.size}\n")
        fh.write(f"BMR_pre_reconciliation={pre_rate:.6f}  ({pre_diff}/{pre_n})\n")
        fh.write(f"BMR_post_reconciliation={post_rate:.6f}  ({post_diff}/{post_n})\n")
        fh.write(f"BMR_post_BCH={bch_rate:.6f}  ({bch_diff}/{bch_n})\n")

    print(f"    [BMR] pre={pre_rate*100:.4f}%  post={post_rate*100:.4f}%  "
          f"post_BCH={bch_rate*100:.4f}%")
    print(f"    [OK] {rep_path.name} written")
    print("Stage 07 complete.\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bob Stage 07 - BMR evaluation")
    parser.add_argument("--exp", required=True, choices=["exp1", "exp2", "exp3"])
    args = parser.parse_args(argv)
    try:
        run(args.exp)
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
