"""
01_alice_distribution_adjustment.py
-----------------------------------
Stage 01 of the KeyCome Group-Key SKG pipeline (Alice).

Reads doss_alice.csv for the chosen experiment.

Column priority (per link):
  1. mtd_delta_XX  — MTD-DOSS: already filtered by Reciprocity Guard
                     NaN = sample contaminated by Eve → DROPPED (not filled)
  2. delta_XX      — raw Mesh-DOSS: fallback if MTD column absent

Applies moving-average smoothing (window=150), zero-centering, and
N×N Gaussian random projection (seed=42) so Bob/Charlie can reproduce
the same projection from their own reciprocal DOSS measurements.

Outputs:
    channelProb/output/01_alice_adjusted_AB_{exp}.csv
    channelProb/output/01_alice_adjusted_AC_{exp}.csv
"""
from __future__ import annotations
import os

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR
OUTPUT_DIR = BASE_DIR / "output"

MA_WINDOW = 150
RP_SEED = 42  # fallback when no session seed file exists


def load_session_seed(exp: str, output_dir: Path) -> int:
    """Load per-session seed written by Stage 00; fall back to RP_SEED=42."""
    seed_path = output_dir / f"00_session_seed_{exp}.txt"
    if seed_path.is_file():
        try:
            seed = int(seed_path.read_text().strip())
            return seed
        except ValueError:
            pass
    return RP_SEED


def kalman_reciprocity_fill(signal: np.ndarray,
                             alpha: float = 0.95,
                             Q: float = 2.0,
                             R: float = 0.5) -> np.ndarray:
    """1-D Kalman filter to fill NaN gaps in a DOSS timeseries.

    Models RSSI channel as AR(1): h[t] = alpha*h[t-1] + w (w~N(0,Q)).
    Valid MTD samples are used as observations; NaN positions are filled
    via Kalman prediction only — never using raw (possibly adversarial) values.
    """
    n = len(signal)
    filled = signal.copy().astype(float)
    valid = np.where(~np.isnan(signal))[0]
    if len(valid) == 0:
        return np.zeros(n, dtype=float)
    x = float(signal[valid[0]])
    P = R
    for t in range(n):
        x = alpha * x
        P = alpha**2 * P + Q
        if not np.isnan(signal[t]):
            K = P / (P + R)
            x = x + K * (float(signal[t]) - x)
            P = (1 - K) * P
            filled[t] = signal[t]
        else:
            filled[t] = x
    return filled


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.astype(float)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x.astype(float), kernel, mode="same")


def random_projection(centered: np.ndarray, seed: int = RP_SEED) -> np.ndarray:
    n = centered.shape[0]
    rng = np.random.default_rng(seed)
    pi_matrix = rng.standard_normal(size=(n, n))
    return centered @ pi_matrix


def load_doss_column(df: pd.DataFrame, link: str, csv_path: Path) -> np.ndarray:
    """Load MTD-DOSS (preferred) or raw Mesh-DOSS for a given link.

    NaN positions (attack-contaminated or non-reciprocal samples) are
    reconstructed via BKRR (Kalman Reciprocity Reconstruction) instead of
    being dropped, preserving the full timeseries length for ADMM.
    """
    mtd_col = f"mtd_delta_{link}"
    raw_col = f"delta_{link}"

    disable_bkrr = os.environ.get("DISABLE_BKRR") == "1"
    if mtd_col in df.columns:
        raw = df[mtd_col].to_numpy(dtype=float)
        n_nan = int(np.isnan(raw).sum())
        n_valid = int(np.sum(~np.isnan(raw)))
        if disable_bkrr:
            samples = raw[~np.isnan(raw)]
            source = "MTD-DOSS (BKRR disabled, NaN dropped)"
            print(f"    [{link}] BKRR DISABLED: {n_nan} NaN dropped "
                  f"(N: {n_valid})")
        else:
            source = "MTD-DOSS+BKRR"
            if n_nan > 0:
                samples = kalman_reciprocity_fill(raw)
                print(f"    [{link}] BKRR: {n_nan} NaN reconstructed via Kalman "
                      f"(N: {n_valid} → {len(samples)})")
            else:
                samples = raw
                print(f"    [{link}] {n_nan} NaN — no reconstruction needed")
    elif raw_col in df.columns:
        raw = df[raw_col].to_numpy(dtype=float)
        if disable_bkrr:
            samples = raw[~np.isnan(raw)]
            source = "Mesh-DOSS (BKRR disabled, NaN dropped)"
            print(f"    [{link}] WARNING: MTD column absent + BKRR DISABLED — using raw {raw_col} with dropna")
        else:
            samples = kalman_reciprocity_fill(raw)
            source = "Mesh-DOSS+BKRR (MTD column absent)"
            print(f"    [{link}] WARNING: MTD column absent — using raw {raw_col} with BKRR")
    else:
        raise KeyError(f"Neither '{mtd_col}' nor '{raw_col}' found in {csv_path}")

    if len(samples) < MA_WINDOW:
        raise ValueError(
            f"[{link}] Only {len(samples)} samples after BKRR "
            f"(need ≥ {MA_WINDOW}). Experiment may be too short."
        )
    print(f"    [{link}] source={source} | N={len(samples)}")
    return samples


def adjust_channel(samples: np.ndarray, label: str, seed: int = RP_SEED) -> np.ndarray:
    smoothed = moving_average(samples, MA_WINDOW)
    centered = smoothed - np.mean(smoothed)
    projected = random_projection(centered, seed)
    print(f"    [{label}] mean(raw)={np.mean(samples):.4f} | mean(centered)={np.mean(centered):.6f} | seed={seed}")
    return projected


def run(exp: str) -> None:
    csv_path = INPUT_DIR / exp / "doss_alice.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    print(f"--- Stage 01: Alice Distribution Adjustment ({exp}) ---")
    df = pd.read_csv(csv_path)

    seed = load_session_seed(exp, OUTPUT_DIR)

    delta_ab = load_doss_column(df, "AB", csv_path)
    delta_ac = load_doss_column(df, "AC", csv_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    s_ab = adjust_channel(delta_ab, "AB", seed)
    s_ac = adjust_channel(delta_ac, "AC", seed)

    out_ab = OUTPUT_DIR / f"01_alice_adjusted_AB_{exp}.csv"
    out_ac = OUTPUT_DIR / f"01_alice_adjusted_AC_{exp}.csv"
    pd.DataFrame({"adjusted_rssi": s_ab}).to_csv(out_ab, index=False)
    pd.DataFrame({"adjusted_rssi": s_ac}).to_csv(out_ac, index=False)

    print(f"    [OK] Saved {out_ab.name} ({len(s_ab)} samples) and {out_ac.name} ({len(s_ac)} samples)")
    print("Stage 01 complete.\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Alice Stage 01 - distribution adjustment")
    parser.add_argument("--exp", required=True, choices=["exp1", "exp2", "exp3"],
                        help="Experiment folder under channelProb/")
    args = parser.parse_args(argv)
    try:
        run(args.exp)
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
