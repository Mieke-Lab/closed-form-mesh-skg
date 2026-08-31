"""
08_charlie_privacy_amplification.py
-----------------------------------
Stage 08 of the KeyCome Group-Key SKG pipeline (Charlie side).

Apply a Toeplitz universal-hash function (m=256, seed=42) on the
reconciled K_group_C to produce the final 256-bit shared key K_final_C.
The seed must match Alice and Bob so that, when K_group_A == K_group_B
== K_group_C, all three parties derive the same K_final.

Output:
    channelProb/output/08_charlie_final_key_{exp}.txt   (hex string)

CLI: python 08_charlie_privacy_amplification.py --exp exp1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

HASH_SEED = 42  # fallback when no session seed file exists


def load_session_seed(exp: str, output_dir: Path) -> int:
    seed_path = output_dir / f"00_session_seed_{exp}.txt"
    if seed_path.is_file():
        try:
            return int(seed_path.read_text().strip())
        except ValueError:
            pass
    return HASH_SEED
HASH_OUTPUT_BITS = 256


def toeplitz_universal_hash(key_bits: np.ndarray, seed: int = HASH_SEED,
                            output_bits: int = HASH_OUTPUT_BITS) -> np.ndarray:
    n = len(key_bits)
    m = output_bits
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, size=(m + n - 1))
    row_idx = np.arange(m).reshape(-1, 1)
    col_idx = np.arange(n).reshape(1, -1)
    T = t[row_idx - col_idx + (n - 1)]
    return (T.dot(key_bits.astype(int)) % 2).astype(int)


def bits_to_hex(bits: np.ndarray) -> str:
    chars = []
    for i in range(0, len(bits), 4):
        nibble = bits[i:i + 4]
        if len(nibble) < 4:
            nibble = np.concatenate([nibble, np.zeros(4 - len(nibble), dtype=int)])
        val = int(nibble[0]) * 8 + int(nibble[1]) * 4 + int(nibble[2]) * 2 + int(nibble[3])
        chars.append(f"{val:x}")
    return "".join(chars)


def run(exp: str) -> None:
    print(f"--- Stage 08: Charlie Privacy Amplification ({exp}) ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    recon = OUTPUT_DIR / f"06_charlie_reconciled_key_{exp}.csv"
    fallback = OUTPUT_DIR / f"05_charlie_recovered_key_{exp}.csv"
    if recon.is_file():
        src = recon
        label = "K_group_C' (reconciled)"
    elif fallback.is_file():
        src = fallback
        label = "K_group_C (no reconciliation available)"
    else:
        raise FileNotFoundError(
            f"Need either {recon.name} or {fallback.name} to derive final key."
        )

    df = pd.read_csv(src)
    real = df["Real"].to_numpy(dtype=int)
    imag = df["Imag"].to_numpy(dtype=int)
    bits = np.concatenate([real, imag])
    print(f"1/2 Source: {src.name} ({label}, {len(bits)} bits).")

    print(f"2/2 Toeplitz hash (m={HASH_OUTPUT_BITS}, seed={HASH_SEED})...")
    seed = load_session_seed(exp, OUTPUT_DIR)
    print(f"2/2 Toeplitz hash (m={HASH_OUTPUT_BITS}, seed={seed})...")
    hashed = toeplitz_universal_hash(bits, seed=seed)
    hex_key = bits_to_hex(hashed)

    out = OUTPUT_DIR / f"08_charlie_final_key_{exp}.txt"
    with out.open("w") as fh:
        fh.write(hex_key + "\n")

    key_dir = BASE_DIR / "key"
    key_dir.mkdir(parents=True, exist_ok=True)
    (key_dir / f"key_charlie_{exp}.txt").write_text(hex_key + "\n")
    print(f"    [KEY] also saved -> {key_dir}/key_charlie_{exp}.txt")

    print("\n==================================================")
    print(f"   Charlie Final 256-bit Shared Key ({exp})")
    print("==================================================")
    print(f"   {hex_key}")
    print("==================================================")
    print(f"[OK] saved to {out}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Charlie Stage 08 - privacy amplification")
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
