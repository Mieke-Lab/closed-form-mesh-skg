"""
02_alice_channel_obfuscation.py
--------------------------------
Stage 02 of the KeyCome Group-Key SKG pipeline (Alice).

For each adjusted channel (AB and AC) build the first row of a circulant
obfuscation matrix Pa using FFT equilibration:
    1. eigenvalues = FFT(first_col), where first_col is the circulant column
       induced by the adjusted vector.
    2. Filter |eigenvalues| with a 7-tap median + 7-tap uniform mean.
    3. Reconstruct eigenvalues with original phase and filtered magnitude.
    4. IFFT -> first column -> derive first row of the circulant matrix.

Storing only the first row is enough because Pa is fully determined by it,
and downstream encapsulation uses O(N log N) FFT convolution.

Outputs:
    channelProb/output/02_alice_matrix_AB_{exp}.csv
    channelProb/output/02_alice_matrix_AC_{exp}.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import medfilt


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

MEDIAN_WINDOW = 7
MEAN_WINDOW = 7


def build_first_row(adjusted: np.ndarray) -> np.ndarray:
    """Equilibrate the circulant matrix and return its first row."""
    n = adjusted.shape[0]
    if n == 0:
        raise ValueError("Adjusted RSSI vector is empty.")

    # Circulant column induced by the adjusted vector: c[k] = s[(-k) mod N]
    first_col = np.concatenate(([adjusted[0]], adjusted[-1:0:-1]))
    eigenvalues = np.fft.fft(first_col)

    sv = np.abs(eigenvalues)
    sv_hat = medfilt(sv, kernel_size=MEDIAN_WINDOW)
    sv_tilde = uniform_filter1d(sv_hat, size=MEAN_WINDOW, mode="constant", cval=0.0)

    phase = np.angle(eigenvalues)
    eigenvalues_tilde = sv_tilde * np.exp(1j * phase)

    first_col_tilde = np.real(np.fft.ifft(eigenvalues_tilde))
    # Map first column back to first row: r[k] = c[(-k) mod N]
    first_row = np.concatenate(([first_col_tilde[0]], first_col_tilde[-1:0:-1]))
    return first_row


def process_channel(label: str, exp: str) -> None:
    in_path = OUTPUT_DIR / f"01_alice_adjusted_{label}_{exp}.csv"
    if not in_path.is_file():
        raise FileNotFoundError(f"Stage 01 output missing: {in_path}")

    df = pd.read_csv(in_path)
    if "adjusted_rssi" not in df.columns:
        raise KeyError(f"'adjusted_rssi' column missing in {in_path}")

    adjusted = df["adjusted_rssi"].to_numpy(dtype=float)
    first_row = build_first_row(adjusted)

    out_path = OUTPUT_DIR / f"02_alice_matrix_{label}_{exp}.csv"
    pd.DataFrame({"first_row": first_row}).to_csv(out_path, index=False)
    print(f"    [{label}] N={adjusted.shape[0]} -> {out_path.name}")


def run(exp: str) -> None:
    print(f"--- Stage 02: Alice Channel Obfuscation ({exp}) ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    process_channel("AB", exp)
    process_channel("AC", exp)
    print("Stage 02 complete.\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Alice Stage 02 - CSI obfuscation")
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
