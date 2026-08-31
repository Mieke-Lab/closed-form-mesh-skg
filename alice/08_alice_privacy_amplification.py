"""
08_alice_privacy_amplification.py
---------------------------------
Stage 08 of the KeyCome Group-Key SKG pipeline (Alice).

Reads the locally-stored 256-bit K_group and applies a Toeplitz universal hash
(m=256, n=key_length, seed=42) to derive K_final. The same matrix is computed
deterministically by Bob and Charlie so all three nodes converge to the same
final key.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

HASH_OUTPUT_BITS = 256
HASH_SEED = 42  # fallback when no session seed file exists


def load_session_seed(exp: str, output_dir: Path) -> int:
    """Load per-session seed written by Stage 00; fall back to HASH_SEED=42."""
    seed_path = output_dir / f"00_session_seed_{exp}.txt"
    if seed_path.is_file():
        try:
            seed = int(seed_path.read_text().strip())
            return seed
        except ValueError:
            pass
    return HASH_SEED


def toeplitz_hash(key_bits: np.ndarray, output_bits: int = HASH_OUTPUT_BITS,
                  seed: int = HASH_SEED) -> np.ndarray:
    """K_final = (T @ key_bits) mod 2 where T is m x n Toeplitz from `seed`."""
    n = key_bits.shape[0]
    m = output_bits
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, size=(m + n - 1))
    rows = np.arange(m).reshape(-1, 1)
    cols = np.arange(n).reshape(1, -1)
    matrix = t[rows - cols + (n - 1)]
    return (matrix.dot(key_bits.astype(int)) % 2).astype(int)


def bits_to_hex(bits: np.ndarray) -> str:
    if bits.size % 4 != 0:
        # pad to nibble boundary just in case
        pad = 4 - (bits.size % 4)
        bits = np.concatenate([bits, np.zeros(pad, dtype=int)])
    out = []
    for i in range(0, bits.size, 4):
        nibble = bits[i:i + 4]
        val = int(nibble[0]) * 8 + int(nibble[1]) * 4 + int(nibble[2]) * 2 + int(nibble[3])
        out.append(f"{val:x}")
    return "".join(out)


def run(exp: str) -> None:
    print(f"--- Stage 08: Alice Privacy Amplification ({exp}) ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    k_path = OUTPUT_DIR / f"03_alice_K_group_{exp}.csv"
    if not k_path.is_file():
        raise FileNotFoundError(f"K_group missing: {k_path}")

    df = pd.read_csv(k_path)
    bits = np.concatenate([df["real"].astype(int).to_numpy(),
                           df["imag"].astype(int).to_numpy()])
    print(f"    [INPUT] K_group has {bits.size} bits")

    seed = load_session_seed(exp, OUTPUT_DIR)
    print(f"    [SEED] using seed={seed}")
    hashed = toeplitz_hash(bits, seed=seed)
    final_hex = bits_to_hex(hashed)

    out_path = OUTPUT_DIR / f"08_alice_final_key_{exp}.txt"
    out_path.write_text(final_hex + "\n")

    key_dir = BASE_DIR / "key"
    key_dir.mkdir(parents=True, exist_ok=True)
    key_path = key_dir / f"key_alice_{exp}.txt"
    key_path.write_text(final_hex + "\n")

    print("    " + "=" * 50)
    print(f"    K_final ({HASH_OUTPUT_BITS} bits, hex):")
    print(f"    {final_hex}")
    print("    " + "=" * 50)
    print(f"    [OK] saved -> {out_path}")
    print(f"    [KEY] also saved -> {key_path}")
    print("Stage 08 complete.\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Alice Stage 08 - privacy amplification")
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
