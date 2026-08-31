"""
04_bob_preparation.py
---------------------
Stage 04 of the KeyCome Group-Key SKG pipeline (Bob side).

Bob receives the encapsulated package b_AB and the BCH parameter file from
Alice via SCP. If Bob has its own channel measurement (doss_bob.csv with
column delta_BA), it performs the same distribution adjustment Alice ran on
delta_AB (MA window=150, zero-centering, and N x N Gaussian random
projection with seed=42), then builds the first row of the equilibrated
circulant matrix Pb_AB. Otherwise Bob waits for Alice to push an adjusted
RSSI vector via SCP.

Outputs:
    channelProb/output/04_bob_adjusted_BA_{exp}.csv  (adjusted_rssi)
    channelProb/output/04_bob_matrix_BA_{exp}.csv    (first_row of Pb_AB)
    channelProb/output/04_bob_prepared_{exp}.txt     (status report)
"""
from __future__ import annotations
import os

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import medfilt


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR
OUTPUT_DIR = BASE_DIR / "output"

MA_WINDOW = 150
RP_SEED = 42  # fallback when no session seed file exists


def load_session_seed(exp: str, output_dir: Path) -> int:
    seed_path = output_dir / f"00_session_seed_{exp}.txt"
    if seed_path.is_file():
        try:
            return int(seed_path.read_text().strip())
        except ValueError:
            pass
    return RP_SEED
MEDIAN_WINDOW = 7
MEAN_WINDOW = 7


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Symmetric moving-average smoother. Output length matches input."""
    if window <= 1:
        return x.astype(float)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x.astype(float), kernel, mode="same")



def kalman_reciprocity_fill(signal, alpha=0.95, Q=2.0, R=0.5):
    """1-D Kalman filter to fill NaN gaps in a DOSS timeseries (BKRR).
    Valid samples are used as observations; NaN filled via Kalman prediction.
    """
    import numpy as _np
    n = len(signal)
    filled = signal.copy().astype(float)
    valid = _np.where(~_np.isnan(signal))[0]
    if len(valid) == 0:
        return _np.zeros(n, dtype=float)
    x = float(signal[valid[0]])
    P = R
    for t in range(n):
        x = alpha * x
        P = alpha**2 * P + Q
        if not _np.isnan(signal[t]):
            K = P / (P + R)
            x = x + K * (float(signal[t]) - x)
            P = (1 - K) * P
            filled[t] = signal[t]
        else:
            filled[t] = x
    return filled

def random_projection(centered: np.ndarray, seed: int = RP_SEED) -> np.ndarray:
    """Multiply centered vector by an N x N Gaussian matrix derived from `seed`.
    Identical to Alice's projection (same seed) so Bob recovers a coherent matrix.
    """
    n = centered.shape[0]
    rng = np.random.default_rng(seed)
    pi_matrix = rng.standard_normal(size=(n, n))
    return centered @ pi_matrix


def adjust_channel(samples: np.ndarray, label: str, seed: int = RP_SEED) -> np.ndarray:
    smoothed = moving_average(samples, MA_WINDOW)
    centered = smoothed - np.mean(smoothed)
    projected = random_projection(centered, seed)
    print(
        f"    [{label}] N={samples.shape[0]} | mean(raw)={np.mean(samples):.4f}"
        f" | mean(smoothed)={np.mean(smoothed):.4f}"
    )
    return projected


def build_first_row(adjusted: np.ndarray) -> np.ndarray:
    """Equilibrate the circulant matrix Pb_AB and return its first row.
    Mirrors Alice's procedure (FFT magnitude filtering with median+mean).
    """
    n = adjusted.shape[0]
    if n == 0:
        raise ValueError("Adjusted RSSI vector is empty.")

    first_col = np.concatenate(([adjusted[0]], adjusted[-1:0:-1]))
    eigenvalues = np.fft.fft(first_col)

    sv = np.abs(eigenvalues)
    sv_hat = medfilt(sv, kernel_size=MEDIAN_WINDOW)
    sv_tilde = uniform_filter1d(sv_hat, size=MEAN_WINDOW, mode="constant", cval=0.0)

    phase = np.angle(eigenvalues)
    eigenvalues_tilde = sv_tilde * np.exp(1j * phase)

    first_col_tilde = np.real(np.fft.ifft(eigenvalues_tilde))
    first_row = np.concatenate(([first_col_tilde[0]], first_col_tilde[-1:0:-1]))
    return first_row


def find_doss_bob(exp: str) -> Path | None:
    """Locate Bob's channel measurement CSV for this experiment."""
    candidate = INPUT_DIR / exp / "doss_bob.csv"
    if candidate.is_file():
        return candidate
    return None


def load_delta_BA(csv_path: Path) -> np.ndarray:
    """Load BA channel DOSS, applying BKRR to fill NaN gaps.
    Uses MTD-filtered column if present, else raw delta_BA.
    NaN positions (packet loss or non-reciprocal) are Kalman-reconstructed,
    unless DISABLE_BKRR=1 env var is set (then NaN is dropped).
    """
    df = pd.read_csv(csv_path)
    disable_bkrr = os.environ.get("DISABLE_BKRR") == "1"
    if "mtd_delta_BA" in df.columns:
        raw = df["mtd_delta_BA"].to_numpy(dtype=float)
        n_nan = int(np.isnan(raw).sum())
        n_valid = int(np.sum(~np.isnan(raw)))
        if disable_bkrr:
            filled = raw[~np.isnan(raw)]
            print(f"    [BA] BKRR DISABLED: {n_nan} NaN dropped (N: {n_valid})")
            print(f"    [BA] source=MTD-DOSS (BKRR off) | N={len(filled)}")
            return filled
        if n_nan > 0:
            filled = kalman_reciprocity_fill(raw)
            print(f"    [BA] BKRR: {n_nan} NaN reconstructed (N: {n_valid} → {len(filled)})")
        else:
            filled = raw
        print(f"    [BA] source=MTD-DOSS+BKRR | N={len(filled)}")
        return filled
    elif "delta_BA" in df.columns:
        raw = df["delta_BA"].to_numpy(dtype=float)
        n_nan = int(np.isnan(raw).sum())
        if disable_bkrr:
            filled = raw[~np.isnan(raw)]
            print(f"    [BA] BKRR DISABLED: {n_nan} NaN dropped")
            print(f"    [BA] source=Mesh-DOSS (BKRR off) | N={len(filled)}")
            return filled
        if n_nan > 0:
            filled = kalman_reciprocity_fill(raw)
            print(f"    [BA] BKRR: {n_nan} NaN reconstructed (N: {int(np.sum(~np.isnan(raw)))} → {len(filled)})")
        else:
            filled = raw
        print(f"    [BA] source=Mesh-DOSS+BKRR | N={len(filled)}")
        return filled
    raise KeyError(
        f"No suitable delta column in {csv_path}. "
        f"Expected mtd_delta_BA or delta_BA. Available: {list(df.columns)}"
    )


def fallback_adjusted(exp: str) -> Path | None:
    """If doss_bob.csv missing, look for an Alice-pushed adjusted RSSI file."""
    candidate = OUTPUT_DIR / f"04_bob_adjusted_BA_{exp}.csv"
    if candidate.is_file():
        return candidate
    pushed = OUTPUT_DIR / f"01_alice_adjusted_AB_{exp}.csv"
    if pushed.is_file():
        return pushed
    return None


def verify_inputs(exp: str) -> tuple[Path, Path]:
    b_ab_path = OUTPUT_DIR / f"03_alice_b_AB_{exp}.csv"
    bch_params_path = OUTPUT_DIR / f"03_BCH_params_{exp}.csv"
    if not b_ab_path.is_file():
        raise FileNotFoundError(f"Encapsulated package missing: {b_ab_path}")
    if not bch_params_path.is_file():
        raise FileNotFoundError(f"BCH params missing: {bch_params_path}")
    return b_ab_path, bch_params_path


def run(exp: str) -> None:
    print(f"--- Stage 04: Bob Preparation ({exp}) ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    b_ab_path, bch_params_path = verify_inputs(exp)
    bch_df = pd.read_csv(bch_params_path)
    params = dict(zip(bch_df["param"].astype(str), bch_df["value"]))
    n_ab = int(params.get("N_AB", 0))
    print(f"    [BCH] params loaded: BCH({int(params['BCH_N'])},"
          f"{int(params['BCH_K'])},t={int(params['BCH_T'])}), "
          f"L={int(params['L'])}, N_AB={n_ab}")

    # Build Pb_AB
    doss = find_doss_bob(exp)
    if doss is not None:
        delta_ba = load_delta_BA(doss)
        if n_ab and delta_ba.shape[0] != n_ab:
            if delta_ba.shape[0] > n_ab:
                print(f"    [BA] Truncating {delta_ba.shape[0]} -> {n_ab} to match Alice N_AB.")
                delta_ba = delta_ba[:n_ab]
            else:
                print(f"    [BA] Padding {delta_ba.shape[0]} -> {n_ab} to match Alice N_AB.")
                delta_ba = np.pad(delta_ba, (0, n_ab - delta_ba.shape[0]), mode='edge')
        seed = load_session_seed(exp, OUTPUT_DIR)
        adjusted = adjust_channel(delta_ba, "BA", seed)
    else:
        fallback = fallback_adjusted(exp)
        if fallback is None:
            raise FileNotFoundError(
                f"Neither doss_bob.csv nor a fallback adjusted-RSSI file is "
                f"available for {exp}. Place doss_bob.csv at {INPUT_DIR/exp/'doss_bob.csv'} "
                f"or have Alice push an adjusted-RSSI vector to "
                f"{OUTPUT_DIR/f'04_bob_adjusted_BA_{exp}.csv'}."
            )
        df = pd.read_csv(fallback)
        col = "adjusted_rssi" if "adjusted_rssi" in df.columns else df.columns[0]
        adjusted = df[col].to_numpy(dtype=float)
        print(f"    [INFO] Loaded adjusted RSSI fallback {fallback.name} (N={adjusted.size})")

    # Sanity dimension check vs Alice
    if n_ab and adjusted.shape[0] != n_ab:
        print(f"    [WARN] N mismatch with Alice: bob N={adjusted.shape[0]} vs N_AB={n_ab}.")

    # Persist adjusted RSSI for downstream / debug
    adj_path = OUTPUT_DIR / f"04_bob_adjusted_BA_{exp}.csv"
    pd.DataFrame({"adjusted_rssi": adjusted}).to_csv(adj_path, index=False)

    first_row = build_first_row(adjusted)
    mat_path = OUTPUT_DIR / f"04_bob_matrix_BA_{exp}.csv"
    pd.DataFrame({"first_row": first_row}).to_csv(mat_path, index=False)

    status_path = OUTPUT_DIR / f"04_bob_prepared_{exp}.txt"
    with status_path.open("w") as fh:
        fh.write("status=OK\n")
        fh.write(f"exp={exp}\n")
        fh.write(f"N={adjusted.shape[0]}\n")
        fh.write(f"N_AB_alice={n_ab}\n")
        fh.write(f"b_AB={b_ab_path.name}\n")
        fh.write(f"matrix_first_row={mat_path.name}\n")
        fh.write(f"adjusted_rssi={adj_path.name}\n")

    print(f"    [OK] {adj_path.name}, {mat_path.name}, {status_path.name} written")
    print("Stage 04 complete.\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bob Stage 04 - preparation")
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
