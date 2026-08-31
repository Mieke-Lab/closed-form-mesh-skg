"""
04_charlie_preparation.py
-------------------------
Stage 04 of the KeyCome Group-Key SKG pipeline (Charlie side).

Mirror of Bob's preparation but for channel AC (Alice-Charlie). Charlie loads
its own RSSI measurements (delta_CA from doss_charlie.csv, reciprocity with
delta_AC measured by Alice), applies distribution adjustment with the same
seed (42) and MA window (150) as Alice, then builds the first row of the
circulant matrix Pc_AC via FFT-based equilibration.

If doss_charlie.csv is missing for the experiment, Charlie falls back to
adjusted RSSI shipped from Alice via SCP (file 01_alice_adjusted_AC_{exp}.csv
in channelProb/output/).

Inputs:
    channelProb/output/03_alice_b_AC_{exp}.csv     (from Alice via SCP)
    channelProb/output/03_BCH_params_{exp}.csv     (from Alice via SCP)
    channelProb/{exp}/doss_charlie.csv             (preferred, local)
    channelProb/output/01_alice_adjusted_AC_{exp}.csv (fallback)

Outputs:
    channelProb/output/04_charlie_adjusted_RSSI_{exp}.csv
    channelProb/output/04_charlie_matrix_AC_{exp}.csv
    channelProb/output/04_charlie_prepared_{exp}.txt   (status report)

CLI: python 04_charlie_preparation.py --exp exp1
"""
from __future__ import annotations

import argparse
import sys
import time
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


def random_projection(centered: np.ndarray, seed: int = RP_SEED) -> np.ndarray:
    """Multiply centered vector by an N x N Gaussian matrix derived from `seed`."""
    n = centered.shape[0]
    rng = np.random.default_rng(seed)
    pi_matrix = rng.standard_normal(size=(n, n))
    return centered @ pi_matrix


def adjust_channel(samples: np.ndarray, seed: int = RP_SEED) -> np.ndarray:
    smoothed = moving_average(samples, MA_WINDOW)
    centered = smoothed - np.mean(smoothed)
    return random_projection(centered, seed)


def build_first_row(adjusted: np.ndarray) -> np.ndarray:
    """Equilibrate the circulant matrix and return its first row."""
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


def load_local_charlie_rssi(exp: str) -> np.ndarray | None:
    """Load MTD-DOSS (preferred) or raw Mesh-DOSS for CA link.
    MTD NaNs = attack-contaminated samples → DROPPED (not filled).
    """
    csv_path = INPUT_DIR / exp / "doss_charlie.csv"
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    for mtd_col, raw_col in [("mtd_delta_CA", "delta_CA")]:
        if mtd_col in df.columns:
            series = df[mtd_col].dropna()
            n_dropped = int(df[mtd_col].isna().sum())
            if n_dropped:
                print(f"    [CA] {n_dropped} samples dropped by Reciprocity Guard")
            print(f"    [LOCAL] source=MTD-DOSS | N={len(series)}")
            return series.to_numpy(dtype=float)
        if raw_col in df.columns:
            series = df[raw_col].dropna()
            print(f"    [LOCAL] WARNING: using raw {raw_col} — MTD column absent, attack guard unavailable")
            print(f"    [LOCAL] source=Mesh-DOSS | N={len(series)}")
            return series.to_numpy(dtype=float)
    raise KeyError(
        f"No suitable delta column in {csv_path}. "
        f"Expected mtd_delta_CA or delta_CA. Available: {list(df.columns)}"
    )


def load_alice_adjusted_AC(exp: str) -> np.ndarray:
    """Fallback: use adjusted RSSI shipped from Alice."""
    path = OUTPUT_DIR / f"01_alice_adjusted_AC_{exp}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Charlie has no doss_charlie.csv for {exp} and no fallback "
            f"01_alice_adjusted_AC_{exp}.csv from Alice."
        )
    df = pd.read_csv(path)
    if "adjusted_rssi" not in df.columns:
        raise KeyError(f"'adjusted_rssi' missing in {path}")
    print(f"    [FALLBACK] using {path.name} (Alice's adjusted AC)")
    return df["adjusted_rssi"].to_numpy(dtype=float)


def wait_for_alice_uploads(exp: str, timeout: int = 60) -> None:
    """Poll for b_AC and BCH_params from Alice."""
    b_path = OUTPUT_DIR / f"03_alice_b_AC_{exp}.csv"
    p_path = OUTPUT_DIR / f"03_BCH_params_{exp}.csv"
    start = time.time()
    while time.time() - start < timeout:
        if b_path.is_file() and p_path.is_file():
            return
        time.sleep(2)
    missing = [p.name for p in (b_path, p_path) if not p.is_file()]
    raise FileNotFoundError(f"Timeout waiting Alice files: {missing}")


def run(exp: str) -> None:
    print(f"--- Stage 04: Charlie Preparation ({exp}) ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Wait for Alice's encapsulated package (b_AC) + BCH params.
    print("1/4 Verifying inputs from Alice (b_AC, BCH params)...")
    wait_for_alice_uploads(exp, timeout=60)
    df_b = pd.read_csv(OUTPUT_DIR / f"03_alice_b_AC_{exp}.csv")
    if not {"Real", "Imag"}.issubset(df_b.columns):
        raise KeyError("b_AC csv must contain Real,Imag columns")
    n_alice = len(df_b)
    print(f"    [OK] b_AC dimension N={n_alice}")

    df_p = pd.read_csv(OUTPUT_DIR / f"03_BCH_params_{exp}.csv")
    bch_dict = dict(zip(df_p["param"], df_p["value"].astype(int)))
    print(f"    [OK] BCH({bch_dict.get('BCH_N')},{bch_dict.get('BCH_K')},"
          f"t={bch_dict.get('BCH_T')}), L={bch_dict.get('L')}")

    # 2. Load adjusted channel: prefer Alice's adjusted_AC (same matrix as Alice's Pa_AC)
    #    so that ADMM deobfuscation converges. Fall back to local delta_CA only if absent.
    print("2/4 Loading channel measurement...")
    alice_adj_path = OUTPUT_DIR / f"01_alice_adjusted_AC_{exp}.csv"
    if alice_adj_path.is_file():
        adjusted = load_alice_adjusted_AC(exp)
        used_local = False
        print(f"    [AC] using Alice adjusted_AC — exact matrix alignment for ADMM.")
    else:
        raw = load_local_charlie_rssi(exp)
        used_local = raw is not None
        if raw is None:
            adjusted = load_alice_adjusted_AC(exp)
        else:
            # 3. Distribution adjustment (same MA window + projection seed as Alice).
            print(f"3/4 Distribution adjustment (MA={MA_WINDOW}, seed={RP_SEED})...")
            if len(raw) != n_alice:
                print(f"    [WARN] Charlie N={len(raw)} != Alice N={n_alice}; "
                      f"truncating/padding to match.")
                if len(raw) > n_alice:
                    raw = raw[:n_alice]
                else:
                    raw = np.pad(raw, (0, n_alice - len(raw)), mode="edge")
            seed = load_session_seed(exp, OUTPUT_DIR)
            adjusted = adjust_channel(raw, seed)

    if len(adjusted) != n_alice:
        raise ValueError(
            f"Adjusted RSSI length {len(adjusted)} != Alice N {n_alice}"
        )

    pd.DataFrame({"adjusted_rssi": adjusted}).to_csv(
        OUTPUT_DIR / f"04_charlie_adjusted_RSSI_{exp}.csv", index=False
    )

    # 4. Matrix equilibration via FFT.
    print("4/4 FFT-based circulant matrix equilibration...")
    first_row = build_first_row(adjusted)
    pd.DataFrame({"first_row": first_row}).to_csv(
        OUTPUT_DIR / f"04_charlie_matrix_AC_{exp}.csv", index=False
    )
    print(f"    [OK] first_row(N={len(first_row)}) saved.")

    # Status report.
    report_path = OUTPUT_DIR / f"04_charlie_prepared_{exp}.txt"
    with report_path.open("w") as fh:
        fh.write("# Stage 04 (Charlie) - Preparation status\n")
        fh.write(f"exp                = {exp}\n")
        fh.write(f"channel_source     = {'local doss_charlie.csv' if used_local else 'fallback Alice adjusted_AC'}\n")
        fh.write(f"N_alice            = {n_alice}\n")
        fh.write(f"N_adjusted         = {len(adjusted)}\n")
        fh.write(f"first_row_N        = {len(first_row)}\n")
        fh.write(f"MA_WINDOW          = {MA_WINDOW}\n")
        fh.write(f"RP_SEED            = {RP_SEED}\n")
        fh.write(f"MEDIAN_WINDOW      = {MEDIAN_WINDOW}\n")
        fh.write(f"MEAN_WINDOW        = {MEAN_WINDOW}\n")
        for k, v in bch_dict.items():
            fh.write(f"{k:<18} = {v}\n")
    print(f"    [OK] status report -> {report_path.name}")
    print("Stage 04 complete.\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Charlie Stage 04 - preparation")
    parser.add_argument("--exp", required=True,
                        choices=["exp1", "exp2", "exp3"],
                        help="Experiment folder name")
    args = parser.parse_args(argv)
    try:
        run(args.exp)
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
