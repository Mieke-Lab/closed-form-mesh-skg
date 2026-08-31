"""
03_alice_key_encoding.py
------------------------
Stage 03 of the KeyCome Group-Key SKG pipeline (Alice).

Generates a single 256-bit group key K_group via os.urandom, encodes it with
BCH(63,30,t=6), interleaves and zero-pads to 2N bits, then encapsulates the
result independently for Bob and Charlie using FFT-based circulant
matrix-vector products derived from the AB and AC obfuscation matrices.

The encapsulated packages b_AB, b_AC and the BCH parameter file are then SCP-ed
to Bob and Charlie respectively.

Outputs (local):
    channelProb/output/03_alice_K_group_{exp}.csv      (Real, Imag bits)
    channelProb/output/03_alice_b_AB_{exp}.csv         (Real, Imag floats)
    channelProb/output/03_alice_b_AC_{exp}.csv         (Real, Imag floats)
    channelProb/output/03_BCH_params_{exp}.csv
"""
from __future__ import annotations
import os

import argparse
import math
import subprocess
import sys
from pathlib import Path

import galois
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

# Network targets
BOB_HOST = os.environ.get("BOB_HOST", "bob@10.0.0.2")
BOB_PASS = os.environ.get("BOB_PASS", "")
BOB_DEST = "/home/bob/secret-key-generation-collection/magisterSeries/2026/IES_basedPTM_No3/channelProb/output"

CHARLIE_HOST = os.environ.get("CHARLIE_HOST", "charlie@10.0.0.3")
CHARLIE_PASS = os.environ.get("CHARLIE_PASS", "")
CHARLIE_DEST = "/home/charlie/secret-key-generation-collection/magisterSeries/2026/IES_basedPTM_No3/channelProb/output"

# BCH(63, 30, t=6): 9 blocks cover 256-bit key with margin.
BCH_N = 63
BCH_K = 30
BCH_T = 6
KEY_BITS = 256
KEY_HALF = KEY_BITS // 2  # 128 real + 128 imag


def generate_group_key() -> tuple[np.ndarray, np.ndarray]:
    """256-bit group key via os.urandom, split as (real_bits, imag_bits)."""
    raw = os.urandom(KEY_BITS // 8)  # 32 bytes
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    return bits[:KEY_HALF].astype(int), bits[KEY_HALF:].astype(int)


def bch_encode_interleave(k_group: np.ndarray) -> tuple[np.ndarray, int, int]:
    """BCH(63,30) encode K_group block-wise, interleave, return encoded bits."""
    bch = galois.BCH(BCH_N, BCH_K)
    n_blocks = math.ceil(KEY_BITS / BCH_K)
    total_encoded = n_blocks * BCH_N

    encoded_blocks = []
    for b in range(n_blocks):
        start = b * BCH_K
        end = min(start + BCH_K, KEY_BITS)
        block = k_group[start:end]
        if len(block) < BCH_K:
            block = np.concatenate([block, np.zeros(BCH_K - len(block), dtype=int)])
        codeword = bch.encode(galois.GF2(block))
        encoded_blocks.append(np.array(codeword, dtype=int))

    all_encoded = np.concatenate(encoded_blocks)  # shape (n_blocks * BCH_N,)
    # Interleave: write row-by-row, read column-by-column to spread errors.
    interleaved = all_encoded.reshape(n_blocks, BCH_N).T.flatten()
    return interleaved, n_blocks, total_encoded


def encapsulate(first_row: np.ndarray, real_bits: np.ndarray,
                imag_bits: np.ndarray) -> np.ndarray:
    """b = circulant(first_row) @ x where x = real + j*imag, via FFT convolution."""
    fft_row = np.fft.fft(first_row)
    b_real = np.real(np.fft.ifft(fft_row * np.fft.fft(real_bits.astype(float))))
    b_imag = np.real(np.fft.ifft(fft_row * np.fft.fft(imag_bits.astype(float))))
    return b_real + 1j * b_imag


def scp_send(local_files: list[Path], host: str, password: str, dest: str) -> bool:
    cmd = [
        "sshpass", "-p", password,
        "scp", "-o", "StrictHostKeyChecking=no",
        *[str(p) for p in local_files],
        f"{host}:{dest}/",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="ignore") if exc.stderr else ""
        print(f"    [WARN] SCP to {host} failed: {stderr.strip()}")
        return False
    except FileNotFoundError as exc:
        print(f"    [WARN] sshpass/scp not available: {exc}")
        return False


def load_first_row(label: str, exp: str) -> np.ndarray:
    path = OUTPUT_DIR / f"02_alice_matrix_{label}_{exp}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Stage 02 output missing: {path}")
    df = pd.read_csv(path)
    if "first_row" not in df.columns:
        raise KeyError(f"'first_row' column missing in {path}")
    return df["first_row"].to_numpy(dtype=float)


def run(exp: str) -> None:
    print(f"--- Stage 03: Alice Key Encoding & Encapsulation ({exp}) ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Group key generation — reuse existing K_group if present (idempotent)
    k_group_path = OUTPUT_DIR / f"03_alice_K_group_{exp}.csv"
    if k_group_path.is_file():
        df_kg = pd.read_csv(k_group_path)
        real_key = df_kg["real"].astype(int).to_numpy()
        imag_key = df_kg["imag"].astype(int).to_numpy()
        print(f"    [KEY] K_group loaded from existing file (stable across re-runs)")
    else:
        real_key, imag_key = generate_group_key()
        pd.DataFrame({"real": real_key, "imag": imag_key}).to_csv(k_group_path, index=False)
        print(f"    [KEY] K_group generated fresh ({KEY_BITS} bits)")

    # 2. BCH encode + interleave + zero-pad
    interleaved, n_blocks, total_encoded = bch_encode_interleave(
        np.concatenate([real_key, imag_key])
    )
    print(f"    [BCH] BCH({BCH_N},{BCH_K},t={BCH_T}), L={n_blocks} blocks "
          f"-> {total_encoded} encoded bits")

    first_row_ab = load_first_row("AB", exp)
    first_row_ac = load_first_row("AC", exp)
    n_ab = first_row_ab.shape[0]
    n_ac = first_row_ac.shape[0]
    if n_ab != n_ac:
        print(f"    [WARN] AB/AC matrix dims differ: N_AB={n_ab} N_AC={n_ac}")

    def pad_to(n: int) -> tuple[np.ndarray, np.ndarray]:
        x_tilde = np.zeros(2 * n, dtype=int)
        x_tilde[:total_encoded] = interleaved
        return x_tilde[:n], x_tilde[n:]

    real_pad_ab, imag_pad_ab = pad_to(n_ab)
    real_pad_ac, imag_pad_ac = pad_to(n_ac)

    # 3. Encapsulation
    b_ab = encapsulate(first_row_ab, real_pad_ab, imag_pad_ab)
    b_ac = encapsulate(first_row_ac, real_pad_ac, imag_pad_ac)

    b_ab_path = OUTPUT_DIR / f"03_alice_b_AB_{exp}.csv"
    b_ac_path = OUTPUT_DIR / f"03_alice_b_AC_{exp}.csv"
    pd.DataFrame({"Real": np.real(b_ab), "Imag": np.imag(b_ab)}).to_csv(b_ab_path, index=False)
    pd.DataFrame({"Real": np.real(b_ac), "Imag": np.imag(b_ac)}).to_csv(b_ac_path, index=False)

    # 4. BCH parameter file (shared by Bob and Charlie)
    bch_params_path = OUTPUT_DIR / f"03_BCH_params_{exp}.csv"
    pd.DataFrame({
        "param": ["BCH_N", "BCH_K", "BCH_T", "L", "KEY_BITS", "N_AB", "N_AC", "total_encoded_bits"],
        "value": [BCH_N, BCH_K, BCH_T, n_blocks, KEY_BITS, n_ab, n_ac, total_encoded],
    }).to_csv(bch_params_path, index=False)
    print(f"    [LOCAL] {b_ab_path.name}, {b_ac_path.name}, {bch_params_path.name} written")

    # 5. SCP dispatch
    seed_path = OUTPUT_DIR / f"00_session_seed_{exp}.txt"

    # CARR: also send b_AC and matrix_AC to Bob for joint ADMM relay constraint
    matrix_ac_path = OUTPUT_DIR / f"02_alice_matrix_AC_{exp}.csv"
    disable_carr = os.environ.get("DISABLE_CARR") == "1"

    if disable_carr:
        print("    [SCP] CARR DISABLED: sending b_AB + BCH params + seed only to Bob ...")
        files_to_bob = [b_ab_path, bch_params_path]
    else:
        print("    [SCP] sending b_AB + b_AC + matrices + BCH params + seed to Bob (CARR) ...")
        files_to_bob = [b_ab_path, b_ac_path, bch_params_path]
        if matrix_ac_path.is_file():
            files_to_bob.append(matrix_ac_path)
    if seed_path.is_file():
        files_to_bob.append(seed_path)
    ok_bob = scp_send(files_to_bob, BOB_HOST, BOB_PASS, BOB_DEST)
    print(f"           {'OK' if ok_bob else 'FAILED'}")

    print("    [SCP] sending b_AC + BCH params + adjusted_AC + seed to Charlie ...")
    adjusted_ac_path = OUTPUT_DIR / f"01_alice_adjusted_AC_{exp}.csv"
    files_to_charlie = [b_ac_path, bch_params_path]
    if adjusted_ac_path.is_file():
        files_to_charlie.append(adjusted_ac_path)
    if seed_path.is_file():
        files_to_charlie.append(seed_path)
    ok_char = scp_send(files_to_charlie, CHARLIE_HOST, CHARLIE_PASS, CHARLIE_DEST)
    print(f"           {'OK' if ok_char else 'FAILED'}")

    print("Stage 03 complete.\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Alice Stage 03 - key encoding & encapsulation")
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
