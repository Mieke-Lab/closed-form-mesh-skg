"""
06_bob_info_reconciliation.py
-----------------------------
Stage 06 of the KeyCome Group-Key SKG pipeline (Bob side).

Bob ships the JSON list of uncertain bit indices to Alice via SCP, then waits
(polling at 2 s, 300 s timeout) for Alice to deposit corrected bits in Bob's
output directory. With the correction, Bob patches its ADMM interleaved
hard-estimate directly (no re-encoding) and BCH-decodes to the reconciled key.
"""
from __future__ import annotations
import os

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import galois
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

ALICE_HOST = os.environ.get("ALICE_HOST", "alice@10.0.0.1")
ALICE_PASS = os.environ.get("ALICE_PASS", "")
ALICE_DEST = (
    "/home/alice/secret-key-generation-collection/magisterSeries/2026/"
    "IES_basedPTM_No3/channelProb/output"
)

POLL_INTERVAL_S = 2.0
POLL_TIMEOUT_S = 300.0


def deinterleave(bits: np.ndarray, n_blocks: int, bch_n: int) -> np.ndarray:
    return bits.reshape(bch_n, n_blocks).T.flatten()


def bch_decode(interleaved_estimate: np.ndarray, n_blocks: int,
               bch_n: int, bch_k: int, key_bits: int) -> np.ndarray:
    bch = galois.BCH(bch_n, bch_k)
    de_il = deinterleave(interleaved_estimate, n_blocks, bch_n)
    decoded = []
    for b in range(n_blocks):
        codeword = de_il[b * bch_n:(b + 1) * bch_n].astype(int)
        msg = bch.decode(galois.GF2(codeword))
        decoded.append(np.array(msg, dtype=int))
    return np.concatenate(decoded)[:key_bits]


def scp_to_alice(local: Path, remote_name: str) -> bool:
    cmd = [
        "sshpass", "-p", ALICE_PASS,
        "scp", "-o", "StrictHostKeyChecking=no",
        str(local),
        f"{ALICE_HOST}:{ALICE_DEST}/{remote_name}",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as exc:
        msg = exc.stderr.decode(errors="ignore") if exc.stderr else ""
        print(f"    [WARN] SCP to Alice failed: {msg.strip()}")
        return False
    except FileNotFoundError as exc:
        print(f"    [WARN] sshpass/scp unavailable: {exc}")
        return False


def wait_for_fix(path: Path, timeout: float, interval: float) -> bool:
    print(f"    [WAIT] polling {path.name} every {interval}s (timeout {timeout}s)")
    start = time.time()
    while time.time() - start < timeout:
        if path.is_file() and path.stat().st_size > 0:
            return True
        time.sleep(interval)
    return False


def copy_recovered_to_reconciled(exp: str, log_path: Path, reason: str) -> None:
    src = OUTPUT_DIR / f"05_bob_recovered_key_{exp}.csv"
    dst = OUTPUT_DIR / f"06_bob_reconciled_key_{exp}.csv"
    shutil.copyfile(src, dst)
    with log_path.open("w") as fh:
        fh.write(f"exp={exp}\n")
        fh.write(f"path={reason}\n")
        fh.write(f"applied_fixes=0\n")
    print(f"    [OK] {reason}: copied {src.name} -> {dst.name}")


def run(exp: str) -> None:
    print(f"--- Stage 06: Bob Information Reconciliation ({exp}) ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    err_path = OUTPUT_DIR / f"05_bob_error_indices_{exp}.json"
    if not err_path.is_file():
        raise FileNotFoundError(f"Stage 05 error-index file missing: {err_path}")

    log_path = OUTPUT_DIR / f"06_bob_reconciliation_log_{exp}.txt"
    err_data = json.loads(err_path.read_text())
    uncertain = err_data.get("indices", [])

    # Always SCP error indices to Alice (even if empty) so Alice Stage 06 is not blocked
    pushed_name = f"06_bob_error_indices_{exp}.json"
    print(f"    [SCP] sending {err_path.name} to Alice as {pushed_name}")
    scp_to_alice(err_path, pushed_name)

    if len(uncertain) == 0:
        copy_recovered_to_reconciled(exp, log_path, reason="no_uncertain_bits")
        print("Stage 06 complete.\n")
        return

    # Push error indices to Alice for correction
    pushed_name = f"06_bob_error_indices_{exp}.json"
    print(f"    [SCP] sending {err_path.name} to Alice as {pushed_name}")
    ok = scp_to_alice(err_path, pushed_name)
    print(f"           {'OK' if ok else 'FAILED (continuing in offline mode)'}")

    fix_path = OUTPUT_DIR / f"06_alice_b_AB_fix_{exp}.csv"
    got_fix = wait_for_fix(fix_path, POLL_TIMEOUT_S, POLL_INTERVAL_S)
    if not got_fix:
        copy_recovered_to_reconciled(exp, log_path, reason="timeout_no_fix_from_alice")
        print("Stage 06 complete (no fix received).\n")
        return

    bch_path = OUTPUT_DIR / f"03_BCH_params_{exp}.csv"
    if not bch_path.is_file():
        raise FileNotFoundError(f"BCH params missing: {bch_path}")
    bch_df = pd.read_csv(bch_path)
    params = dict(zip(bch_df["param"].astype(str), bch_df["value"]))
    bch_n = int(params["BCH_N"])
    bch_k = int(params["BCH_K"])
    n_blocks = int(params["L"])
    key_bits = int(params["KEY_BITS"])
    total_encoded = int(params["total_encoded_bits"])

    fix_df = pd.read_csv(fix_path)

    if "bit" in fix_df.columns and "index" in fix_df.columns:
        # Load the raw interleaved hard estimate saved by Stage 05.
        # Patching Alice's corrections onto this avoids re-encoding errors.
        il_path = OUTPUT_DIR / f"05_bob_interleaved_estimate_{exp}.csv"
        if not il_path.is_file():
            raise FileNotFoundError(f"Interleaved estimate missing: {il_path}")
        interleaved = pd.read_csv(il_path)["bit"].to_numpy(dtype=int)
        if interleaved.size < total_encoded:
            interleaved = np.concatenate([interleaved,
                                          np.zeros(total_encoded - interleaved.size,
                                                   dtype=int)])
        interleaved = interleaved[:total_encoded].copy()
        for _, row in fix_df.iterrows():
            idx = int(row["index"])
            bit = int(row["bit"]) & 1
            if 0 <= idx < total_encoded:
                interleaved[idx] = bit
        applied = len(fix_df)
    else:
        col = fix_df.columns[0]
        interleaved = fix_df[col].to_numpy(dtype=int)[:total_encoded]
        applied = total_encoded

    decoded = bch_decode(interleaved, n_blocks, bch_n, bch_k, key_bits)
    real_bits = decoded[: key_bits // 2].astype(int)
    imag_bits = decoded[key_bits // 2:].astype(int)
    rec_path = OUTPUT_DIR / f"06_bob_reconciled_key_{exp}.csv"
    pd.DataFrame({"real": real_bits, "imag": imag_bits}).to_csv(rec_path, index=False)

    with log_path.open("w") as fh:
        fh.write(f"exp={exp}\n")
        fh.write(f"path=fix_applied\n")
        fh.write(f"applied_fixes={applied}\n")
        fh.write(f"uncertain_reported={len(uncertain)}\n")
    print(f"    [OK] applied {applied} fix entries -> {rec_path.name}")
    print("Stage 06 complete.\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bob Stage 06 - info reconciliation")
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
