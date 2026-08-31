"""
00_session_seed.py
------------------
Stage 00 of the KeyCome Group-Key SKG pipeline (Alice).

Derives a per-session integer seed from the raw DOSS channel measurements
using SHA-256, then distributes the seed to Bob and Charlie via SCP.

The seed is used to randomise the Gaussian projection matrix (Stage 01)
and the Toeplitz privacy-amplification matrix (Stage 08) so that each
physical measurement session yields different matrix realisations, preventing
an adversary who knows a past matrix from predicting the next one.

This is NOT noise perturbation — the channel samples are not modified.
The seed only controls which random matrices are drawn.

Output (local):
    channelProb/output/00_session_seed_{exp}.txt    (single integer)
"""
from __future__ import annotations
import os

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

BOB_HOST = os.environ.get("BOB_HOST", "bob@10.0.0.2")
BOB_PASS = os.environ.get("BOB_PASS", "")
BOB_DEST = "/home/bob/secret-key-generation-collection/magisterSeries/2026/IES_basedPTM_No3/channelProb/output"

CHARLIE_HOST = os.environ.get("CHARLIE_HOST", "charlie@10.0.0.3")
CHARLIE_PASS = os.environ.get("CHARLIE_PASS", "")
CHARLIE_DEST = "/home/charlie/secret-key-generation-collection/magisterSeries/2026/IES_basedPTM_No3/channelProb/output"

N_SAMPLES = 50  # number of raw DOSS samples used for seed derivation


def derive_session_seed(raw_doss: np.ndarray, n: int = N_SAMPLES) -> int:
    """SHA-256 of the first n raw DOSS float64 values → 32-bit integer seed."""
    snippet = raw_doss[:n].astype(np.float64)
    digest = hashlib.sha256(snippet.tobytes()).hexdigest()
    return int(digest[:8], 16)  # first 4 bytes → 32-bit unsigned int


def load_raw_doss(exp: str) -> np.ndarray:
    csv_path = BASE_DIR / exp / "doss_alice.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"doss_alice.csv not found: {csv_path}")
    df = pd.read_csv(csv_path)
    for col in ("delta_AB", "mtd_delta_AB"):
        if col in df.columns:
            vals = df[col].dropna().to_numpy(dtype=np.float64)
            if len(vals) >= N_SAMPLES:
                return vals
    raise KeyError(
        f"Need at least {N_SAMPLES} non-NaN values in delta_AB or mtd_delta_AB in {csv_path}"
    )


def scp_send(local_file: Path, host: str, password: str, dest: str) -> bool:
    cmd = [
        "sshpass", "-p", password,
        "scp", "-o", "StrictHostKeyChecking=no",
        str(local_file), f"{host}:{dest}/",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="ignore") if exc.stderr else ""
        print(f"    [WARN] SCP to {host} failed: {stderr.strip()}")
        return False
    except FileNotFoundError:
        print(f"    [WARN] sshpass/scp not available")
        return False


def run(exp: str) -> None:
    print(f"--- Stage 00: Session Seed Derivation ({exp}) ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_doss(exp)
    seed = derive_session_seed(raw)
    print(f"    [SEED] derived from delta_AB[:50] -> seed={seed} (0x{seed:08x})")

    seed_path = OUTPUT_DIR / f"00_session_seed_{exp}.txt"
    seed_path.write_text(str(seed) + "\n")
    print(f"    [LOCAL] {seed_path.name} written")

    print("    [SCP] distributing seed to Bob ...")
    ok_bob = scp_send(seed_path, BOB_HOST, BOB_PASS, BOB_DEST)
    print(f"           {'OK' if ok_bob else 'FAILED (Bob will fall back to seed=42)'}")

    print("    [SCP] distributing seed to Charlie ...")
    ok_charlie = scp_send(seed_path, CHARLIE_HOST, CHARLIE_PASS, CHARLIE_DEST)
    print(f"           {'OK' if ok_charlie else 'FAILED (Charlie will fall back to seed=42)'}")

    print("Stage 00 complete.\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Alice Stage 00 - session seed derivation")
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
