"""
07_charlie_bmr_evaluation.py
----------------------------
Stage 07 of the KeyCome Group-Key SKG pipeline (Charlie side).

Compute BMR pre-/post-reconciliation against Alice's ground-truth K_group.
The ground-truth K_group is fetched from Alice via SCP (file
03_alice_K_group_{exp}.csv on Alice). The script reports:

  * BMR pre-BCH    : x_tilde_C (rounded, info region) vs interleaved Alice key
                     reconstructed locally as soft-encoded reference.
  * BMR pre-recon  : K_group_C  (stage 05) vs K_group_A (Alice).
  * BMR post-recon : K_group_C' (stage 06) vs K_group_A (Alice).

Outputs:
    channelProb/output/07_charlie_bmr_report_{exp}.txt

CLI: python 07_charlie_bmr_evaluation.py --exp exp1
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
ALICE_DIR = ("/home/alice/secret-key-generation-collection/"
             "magisterSeries/2026/IES_basedPTM_No3/channelProb/output")


def fetch_alice_k_group(exp: str) -> Path:
    remote = f"03_alice_K_group_{exp}.csv"
    local = OUTPUT_DIR / f"EVAL_{remote}"
    cmd = [
        "sshpass", "-p", ALICE_PASS,
        "scp", "-o", "StrictHostKeyChecking=no",
        f"{ALICE_HOST}:{ALICE_DIR}/{remote}",
        str(local),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return local


def bmr(a: np.ndarray, b: np.ndarray) -> tuple[float, int, int]:
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    err = int(np.sum(a != b))
    return (err / n) * 100.0 if n else 0.0, err, n


def run(exp: str) -> None:
    print(f"--- Stage 07: Charlie BMR Evaluation ({exp}) ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("1/4 Fetching Alice ground-truth K_group...")
    alice_path = fetch_alice_k_group(exp)
    df_a = pd.read_csv(alice_path)
    K_a = np.concatenate([df_a["real"].to_numpy(dtype=int),
                          df_a["imag"].to_numpy(dtype=int)])
    print(f"    [OK] K_group_A loaded ({len(K_a)} bits).")

    print("2/4 Loading Charlie keys...")
    df_pre = pd.read_csv(OUTPUT_DIR / f"05_charlie_recovered_key_{exp}.csv")
    K_pre = np.concatenate([df_pre["Real"].to_numpy(dtype=int),
                            df_pre["Imag"].to_numpy(dtype=int)])
    df_post = pd.read_csv(OUTPUT_DIR / f"06_charlie_reconciled_key_{exp}.csv")
    K_post = np.concatenate([df_post["Real"].to_numpy(dtype=int),
                             df_post["Imag"].to_numpy(dtype=int)])

    print("3/4 Computing BMRs...")
    bmr_pre,  err_pre,  n_pre  = bmr(K_a, K_pre)
    bmr_post, err_post, n_post = bmr(K_a, K_post)

    # BMR pre-BCH: compare encoded x_tilde region vs Alice rounded info bits.
    df_est = pd.read_csv(OUTPUT_DIR / f"05_charlie_estimated_key_{exp}.csv")
    df_p = pd.read_csv(OUTPUT_DIR / f"03_BCH_params_{exp}.csv")
    bch_dict = dict(zip(df_p["param"], df_p["value"].astype(int)))
    total_encoded_bits = bch_dict["total_encoded_bits"]

    # Reconstruct Alice's interleaved encoded info region locally for fair compare.
    import galois
    BCH_N, BCH_K, KEY_BITS = bch_dict["BCH_N"], bch_dict["BCH_K"], bch_dict["KEY_BITS"]
    L = bch_dict["L"]
    bch = galois.BCH(BCH_N, BCH_K)

    def encode_interleave(k_bits: np.ndarray) -> np.ndarray:
        n_blocks = L
        blocks = []
        for b in range(n_blocks):
            start = b * BCH_K
            end = min(start + BCH_K, KEY_BITS)
            block = k_bits[start:end]
            if len(block) < BCH_K:
                block = np.concatenate([block, np.zeros(BCH_K - len(block), dtype=int)])
            cw = bch.encode(galois.GF2(block))
            blocks.append(np.array(cw, dtype=int))
        all_enc = np.concatenate(blocks)
        return all_enc.reshape(n_blocks, BCH_N).T.flatten()

    alice_info = encode_interleave(K_a)
    charlie_x_info = np.concatenate([
        df_est["Real_Rounded"].to_numpy(dtype=int),
        df_est["Imag_Rounded"].to_numpy(dtype=int),
    ])[:total_encoded_bits]
    bmr_encoded, err_enc, n_enc = bmr(alice_info, charlie_x_info)

    print("4/4 Writing report...")
    report = OUTPUT_DIR / f"07_charlie_bmr_report_{exp}.txt"
    with report.open("w") as fh:
        fh.write("# Stage 07 (Charlie) - BMR evaluation\n")
        fh.write(f"exp                       = {exp}\n")
        fh.write(f"BMR pre-BCH (encoded info): {bmr_encoded:.4f} %  ({err_enc}/{n_enc})\n")
        fh.write(f"BMR pre-reconciliation    : {bmr_pre:.4f} %  ({err_pre}/{n_pre})\n")
        fh.write(f"BMR post-reconciliation   : {bmr_post:.4f} %  ({err_post}/{n_post})\n")
        fh.write(f"K_group bits              = {len(K_a)}\n")
        fh.write(f"match_pre  = {'YES' if err_pre  == 0 else 'NO'}\n")
        fh.write(f"match_post = {'YES' if err_post == 0 else 'NO'}\n")

    print(f"    [OK] {report.name}")
    print("\n==============================================")
    print(f"  Charlie BMR ({exp})")
    print("==============================================")
    print(f"  pre-BCH    : {bmr_encoded:.4f} % ({err_enc}/{n_enc})")
    print(f"  pre-recon  : {bmr_pre:.4f} % ({err_pre}/{n_pre})")
    print(f"  post-recon : {bmr_post:.4f} % ({err_post}/{n_post})")
    print("==============================================\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Charlie Stage 07 - BMR evaluation")
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
