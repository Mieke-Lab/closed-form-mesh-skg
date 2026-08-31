"""
08_bob_privacy_amplification.py
-------------------------------
Stage 08 of the KeyCome Group-Key SKG pipeline (Bob side).

Compresses the 256-bit reconciled key with a Toeplitz hash (m=256, n=256,
seed=42) so that Bob and Alice derive identical final keys whenever
reconciliation is successful. Uses T[i,j] = t[i-j+(n-1)] (same as Alice/Charlie).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

PA_M = 256
PA_N = 256
PA_SEED = 42  # fallback when no session seed file exists


def load_session_seed(exp: str, output_dir: Path) -> int:
    seed_path = output_dir / f"00_session_seed_{exp}.txt"
    if seed_path.is_file():
        try:
            return int(seed_path.read_text().strip())
        except ValueError:
            pass
    return PA_SEED


def toeplitz_hash(key_bits: np.ndarray, m: int = PA_M, seed: int = PA_SEED) -> np.ndarray:
    n = key_bits.shape[0]
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, size=(m + n - 1))
    row_idx = np.arange(m).reshape(-1, 1)
    col_idx = np.arange(n).reshape(1, -1)
    T = t[row_idx - col_idx + (n - 1)]
    return (T.dot(key_bits.astype(int)) % 2).astype(int)


def bits_to_hex(bits: np.ndarray) -> str:
    chars = []
    for i in range(0, bits.size, 4):
        nibble = bits[i:i + 4]
        if len(nibble) < 4:
            nibble = np.concatenate([nibble, np.zeros(4 - len(nibble), dtype=int)])
        val = int(nibble[0]) * 8 + int(nibble[1]) * 4 + int(nibble[2]) * 2 + int(nibble[3])
        chars.append(f"{val:x}")
    return "".join(chars)


def run(exp: str) -> None:
    print(f"--- Stage 08: Bob Privacy Amplification ({exp}) ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    src = OUTPUT_DIR / f"06_bob_reconciled_key_{exp}.csv"
    if not src.is_file():
        raise FileNotFoundError(f"Stage 06 reconciled key missing: {src}")

    df = pd.read_csv(src)
    key_bits = np.concatenate([
        df["real"].to_numpy(dtype=np.int32),
        df["imag"].to_numpy(dtype=np.int32),
    ])
    if key_bits.size != PA_N:
        if key_bits.size < PA_N:
            key_bits = np.concatenate([key_bits,
                                       np.zeros(PA_N - key_bits.size, dtype=np.int32)])
        else:
            key_bits = key_bits[:PA_N]
        print(f"    [WARN] reconciled length != {PA_N}; padded/truncated to {PA_N}")

    seed = load_session_seed(exp, OUTPUT_DIR)
    print(f"    [SEED] using seed={seed}")
    final_bits = toeplitz_hash(key_bits, seed=seed)
    hex_str = bits_to_hex(final_bits)

    out = OUTPUT_DIR / f"08_bob_final_key_{exp}.txt"
    out.write_text(hex_str + "\n")

    key_dir = BASE_DIR / "key"
    key_dir.mkdir(parents=True, exist_ok=True)
    (key_dir / f"key_bob_{exp}.txt").write_text(hex_str + "\n")
    print(f"    [KEY] also saved -> {key_dir}/key_bob_{exp}.txt")
    print(f"    [PA] m={PA_M} n={PA_N} seed={PA_SEED} -> {len(hex_str)*4} bits")
    print(f"    [KEY] {hex_str}")
    print(f"    [OK] {out.name} written")
    print("Stage 08 complete.\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bob Stage 08 - privacy amplification")
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
