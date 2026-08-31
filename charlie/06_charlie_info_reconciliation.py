"""
06_charlie_info_reconciliation.py
---------------------------------
Stage 06 of the KeyCome Group-Key SKG pipeline (Charlie side).

Alice pulls Charlie's uncertain-bit indices directly (no sshpass on Charlie),
then pushes (index, bit) direct corrections as
06_alice_charlie_sub_encapsulation_{exp}.csv. Charlie patches its interleaved
hard-estimate from Stage 05 with those corrections and BCH-decodes to obtain
the reconciled K_group_C.

Workflow:
    1. Load uncertain indices from 05_charlie_error_indices_{exp}.json.
    2. Wait (poll) until Alice deposits the fix file in local output dir.
    3. Load interleaved hard-estimate from 05_charlie_estimated_key_{exp}.csv.
    4. Patch uncertain positions with Alice's correct bits.
    5. BCH decode -> 06_charlie_reconciled_key_{exp}.csv.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import galois
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

POLL_INTERVAL = 2
POLL_TIMEOUT = 300


def deinterleave(bits: np.ndarray, n_blocks: int, bch_n: int) -> np.ndarray:
    return bits.reshape(bch_n, n_blocks).T.flatten()


def bch_decode(interleaved: np.ndarray, n_blocks: int,
               bch_n: int, bch_k: int, key_bits: int) -> np.ndarray:
    bch = galois.BCH(bch_n, bch_k)
    de_il = deinterleave(interleaved, n_blocks, bch_n)
    decoded = []
    for b in range(n_blocks):
        codeword = de_il[b * bch_n:(b + 1) * bch_n].astype(int)
        try:
            msg = bch.decode(galois.GF2(codeword))
            decoded.append(np.array(msg, dtype=int))
        except Exception:
            decoded.append(codeword[:bch_k].astype(int))
    return np.concatenate(decoded)[:key_bits]


def copy_recovered_to_reconciled(exp: str) -> None:
    src = OUTPUT_DIR / f"05_charlie_recovered_key_{exp}.csv"
    dst = OUTPUT_DIR / f"06_charlie_reconciled_key_{exp}.csv"
    src_df = pd.read_csv(src)
    if "Real" in src_df.columns and "Imag" in src_df.columns:
        pd.DataFrame({"Real": src_df["Real"], "Imag": src_df["Imag"]}).to_csv(dst, index=False)
    else:
        src_df.to_csv(dst, index=False)
    print(f"    [OK] fallback: copied {src.name} -> {dst.name}")


def run(exp: str) -> None:
    print(f"--- Stage 06: Charlie Information Reconciliation ({exp}) ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    indices_path = OUTPUT_DIR / f"05_charlie_error_indices_{exp}.json"
    if not indices_path.is_file():
        raise FileNotFoundError(f"Missing {indices_path}; run stage 05 first.")
    with indices_path.open() as fh:
        error_indices = json.load(fh)
    print(f"1/4 Loaded {len(error_indices)} uncertain indices.")

    if not error_indices:
        copy_recovered_to_reconciled(exp)
        print("Stage 06 complete (no reconciliation needed).\n")
        return

    # 2/3: Alice has already pulled the indices and will push fix file.
    print("2/4 Waiting for Alice to pull indices and push corrections...")

    fix_path = OUTPUT_DIR / f"06_alice_charlie_sub_encapsulation_{exp}.csv"
    print(f"3/4 Polling {fix_path.name} (timeout {POLL_TIMEOUT}s)...")
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        if fix_path.is_file() and fix_path.stat().st_size > 0:
            break
        time.sleep(POLL_INTERVAL)
    else:
        print("    [WARN] Timeout waiting for Alice fix. Falling back to Stage 05 key.")
        copy_recovered_to_reconciled(exp)
        print("Stage 06 complete (timeout fallback).\n")
        return

    fix_df = pd.read_csv(fix_path)

    bch_path = OUTPUT_DIR / f"03_BCH_params_{exp}.csv"
    bch_df = pd.read_csv(bch_path)
    params = dict(zip(bch_df["param"].astype(str), bch_df["value"]))
    bch_n = int(params["BCH_N"])
    bch_k = int(params["BCH_K"])
    n_blocks = int(params["L"])
    key_bits = int(params["KEY_BITS"])
    total_encoded = int(params["total_encoded_bits"])

    # Load interleaved hard estimate from Stage 05 (Real_Rounded[:total_encoded]).
    est_path = OUTPUT_DIR / f"05_charlie_estimated_key_{exp}.csv"
    if not est_path.is_file():
        raise FileNotFoundError(f"Estimated key missing: {est_path}")
    est_df = pd.read_csv(est_path)
    interleaved = est_df["Real_Rounded"].to_numpy(dtype=int)[:total_encoded].copy()

    print("4/4 Patching corrections and BCH-decoding...")
    if "bit" in fix_df.columns and "index" in fix_df.columns:
        applied = 0
        for _, row in fix_df.iterrows():
            idx = int(row["index"])
            bit = int(row["bit"]) & 1
            if 0 <= idx < total_encoded:
                interleaved[idx] = bit
                applied += 1
        print(f"    [OK] patched {applied} bit corrections.")
    else:
        print("    [WARN] Unexpected fix format — skipping patch.")

    decoded = bch_decode(interleaved, n_blocks, bch_n, bch_k, key_bits)
    half = key_bits // 2
    K_real = decoded[:half].astype(int)
    K_imag = decoded[half:].astype(int)

    out_path = OUTPUT_DIR / f"06_charlie_reconciled_key_{exp}.csv"
    pd.DataFrame({"Real": K_real, "Imag": K_imag}).to_csv(out_path, index=False)
    print(f"    [OK] reconciled key -> {out_path.name}")
    print("Stage 06 complete.\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Charlie Stage 06 - info reconciliation")
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
