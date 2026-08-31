"""
12_paper_metrics.py
-------------------
Aggregates pipeline outputs into a paper-ready metrics table.

Reads (for a given exp):
    output/03_BCH_params_{exp}.csv          N_samples, KEY_BITS
    output/05_bob_admm_report_{exp}.txt     wall_seconds, iterations, uncertain_bits
    output/07_bob_bmr_report_{exp}.txt      BMR pre/post reconciliation
    output/10_nist_results_{exp}.csv        NIST PASS/FAIL
    output/08_alice_final_key_{exp}.txt     Alice final key (hex)
    Bob: output/08_bob_final_key_{exp}.txt  Bob final key (hex) for KDR check

Computed metrics:
    BGR   = KEY_BITS / N_samples            (bits per DOSS sample)
    BPS   = KEY_BITS / wall_seconds_admm    (bits per second, ADMM stage)
    KDR   = 0 if keys match else 1          (key disagreement rate)
    NIST  = PASS if all tests pass else FAIL

Output:
    output/12_paper_metrics_{exp}.csv
    ASCII table to stdout
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

BOB_HOST = os.environ.get("BOB_HOST", "bob@10.0.0.2")
BOB_PASS = os.environ.get("BOB_PASS", "")
BOB_REMOTE_OUTPUT = "/home/bob/secret-key-generation-collection/magisterSeries/2026/IES_basedPTM_No3/channelProb/output"


def scp_pull(remote_path: str, local_path: Path) -> bool:
    cmd = [
        "sshpass", "-p", BOB_PASS,
        "scp", "-o", "StrictHostKeyChecking=no",
        f"{BOB_HOST}:{remote_path}", str(local_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def pull_bob_files(exp: str) -> None:
    for fname in (f"07_bob_bmr_report_{exp}.txt",
                  f"05_bob_admm_report_{exp}.txt",
                  f"08_bob_final_key_{exp}.txt"):
        local = OUTPUT_DIR / fname
        ok = scp_pull(f"{BOB_REMOTE_OUTPUT}/{fname}", local)
        print(f"    [SCP] {'OK' if ok else 'FAILED'}: pulled {fname}")


def parse_report(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    d = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            d[k.strip()] = v.strip()
    return d


def run(exp: str) -> None:
    print(f"--- Stage 12: Paper Metrics ({exp}) ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pull_bob_files(exp)

    # BCH params: N_samples
    bch_path = OUTPUT_DIR / f"03_BCH_params_{exp}.csv"
    if bch_path.is_file():
        bch = dict(zip(
            pd.read_csv(bch_path)["param"],
            pd.read_csv(bch_path)["value"].astype(float)
        ))
        n_samples = int(bch.get("N_AB", 0))
        key_bits = int(bch.get("KEY_BITS", 256))
    else:
        n_samples, key_bits = 0, 256
        print("    [WARN] BCH params missing")

    # Total pipeline time (written by 09_alice_orchestrator.py)
    total_time = parse_report(OUTPUT_DIR / f"09_total_time_{exp}.txt")
    total_s = float(total_time.get("total_wall_seconds", 0) or 0)

    # ADMM report: iterations
    admm = parse_report(OUTPUT_DIR / f"05_bob_admm_report_{exp}.txt")
    it_real = int(admm.get("real_iterations", 0) or 0)
    it_imag = int(admm.get("imag_iterations", 0) or 0)
    conf_tol = admm.get("confidence_tol", "adaptive")

    # BMR report
    bmr = parse_report(OUTPUT_DIR / f"07_bob_bmr_report_{exp}.txt")
    bmr_pre = bmr.get("BMR_pre_reconciliation", bmr.get("bmr_pre", "N/A"))
    bmr_post = bmr.get("BMR_post_reconciliation", bmr.get("bmr_post", "0.0000"))

    # NIST results
    nist_path = OUTPUT_DIR / f"10_nist_results_{exp}.csv"
    if nist_path.is_file():
        nist_df = pd.read_csv(nist_path)
        n_pass = (nist_df["result"] == "PASS").sum()
        n_total = len(nist_df)
        nist_status = "PASS" if n_pass == n_total else f"FAIL ({n_pass}/{n_total})"
    else:
        nist_status = "N/A"

    # KDR: compare Alice and Bob final keys
    alice_key_path = OUTPUT_DIR / f"08_alice_final_key_{exp}.txt"
    bob_key_path = OUTPUT_DIR / f"08_bob_final_key_{exp}.txt"
    kdr = "N/A"
    if alice_key_path.is_file() and bob_key_path.is_file():
        k_alice = alice_key_path.read_text().strip()
        k_bob = bob_key_path.read_text().strip()
        kdr = "0%" if k_alice == k_bob else "100%"

    # Eve security metrics (from Stage 11)
    eve_path = OUTPUT_DIR / f"11_eve_results_{exp}.csv"
    eve_kdr_ab = eve_r = "N/A"
    if eve_path.is_file():
        eve_df = pd.read_csv(eve_path)
        if not eve_df.empty:
            eve_kdr_ab = str(eve_df["eve_kdr_via_AB"].iloc[0])
            eve_r      = str(eve_df["pearson_r_AliceEve_AB"].iloc[0])

    # Computed metrics
    bgr = key_bits / n_samples if n_samples > 0 else float("nan")
    kgr = key_bits / total_s if total_s > 0 else float("nan")

    row = {
        "exp": exp,
        "N": n_samples,
        "KEY_BITS": key_bits,
        "BMR_pre": bmr_pre,
        "BMR_post": bmr_post,
        "BGR": f"{bgr:.4f}" if not np.isnan(bgr) else "N/A",
        "KGR": f"{kgr:.3f}" if not np.isnan(kgr) else "N/A",
        "ADMM_iters": f"{it_real},{it_imag}",
        "conf_tol": conf_tol,
        "NIST": nist_status,
        "KDR": kdr,
        "Eve_KDR": eve_kdr_ab,
        "r_AliceEve": eve_r,
    }

    out_path = OUTPUT_DIR / f"12_paper_metrics_{exp}.csv"
    pd.DataFrame([row]).to_csv(out_path, index=False)

    # ASCII table
    print("\n" + "=" * 75)
    print(f"  Paper Metrics — {exp}")
    print("=" * 75)
    for k, v in row.items():
        print(f"  {k:<18} {v}")
    print("=" * 75)
    print(f"    [OK] {out_path.name} written")
    print("Stage 12 complete.\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paper metrics aggregator")
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
