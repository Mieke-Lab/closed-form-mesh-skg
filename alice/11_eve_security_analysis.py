"""
11_eve_security_analysis.py
---------------------------
Security evaluation: simulate Eve's passive eavesdropping attack.

Eve observes channel measurements (eve_approx_AB, eve_approx_AC) from
doss_eve_alice.csv and attempts to recover K_group using the same pipeline
as the legitimate nodes — but with her own (wrong) channel matrix.

Attack model (worst-case assumption):
  - Eve knows the complete pipeline algorithm
  - Eve captures b_AB and b_AC from intercepted SCP traffic
  - Eve builds her own circulant matrix from her passive DOSS measurements
  - Eve runs TRSD deobfuscation with her wrong matrix
  - Eve's recovered key is compared to Alice's true K_group

Expected result: Eve KDR ≈ 50% (random guessing)

Outputs:
    channelProb/output/11_eve_results_{exp}.csv
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import galois
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import medfilt
from scipy.stats import pearsonr


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
EXP_DATA_DIR = BASE_DIR  # exp1/, exp2/, exp3/ subdirs live here

# ── TRSD params (mirror Bob Stage 05) ────────────────────────────────────────
TRSD_ALPHA_REL = 1e-3
MA_WINDOW   = 150
MEDIAN_WIN  = 7
MEAN_WIN    = 7

# ── BCH params ────────────────────────────────────────────────────────────────
BCH_N, BCH_K, BCH_T = 63, 30, 6


# ── Helpers (from Stage 01 / Stage 02) ───────────────────────────────────────

def kalman_fill(signal: np.ndarray, alpha=0.95, Q=2.0, R=0.5) -> np.ndarray:
    filled = signal.copy().astype(float)
    valid  = np.where(~np.isnan(signal))[0]
    if len(valid) == 0:
        return np.zeros(len(signal))
    x, P = float(signal[valid[0]]), R
    for t in range(len(signal)):
        x = alpha * x
        P = alpha**2 * P + Q
        if not np.isnan(signal[t]):
            K = P / (P + R)
            x += K * (float(signal[t]) - x)
            P  = (1 - K) * P
            filled[t] = signal[t]
        else:
            filled[t] = x
    return filled


def build_first_row(adjusted: np.ndarray) -> np.ndarray:
    """Stage 02 CSI obfuscation: equilibrated circulant first row."""
    first_col   = np.concatenate(([adjusted[0]], adjusted[-1:0:-1]))
    eig         = np.fft.fft(first_col)
    sv          = np.abs(eig)
    sv_hat      = medfilt(sv, kernel_size=MEDIAN_WIN)
    sv_tilde    = uniform_filter1d(sv_hat, size=MEAN_WIN, mode="constant", cval=0.0)
    phase       = np.angle(eig)
    eig_tilde   = sv_tilde * np.exp(1j * phase)
    col_tilde   = np.real(np.fft.ifft(eig_tilde))
    return np.concatenate(([col_tilde[0]], col_tilde[-1:0:-1]))


def admm_deobfuscate(first_row: np.ndarray, b_real: np.ndarray,
                     b_imag: np.ndarray) -> np.ndarray:
    """TRSD (Tikhonov-Regularized Spectral Decoding) for Eve attack.

    One-shot closed-form deobfuscation matching the legitimate pipeline.
    Function name kept for back-compat; implementation is now TRSD.
    """
    TRSD_ALPHA_REL = 1e-3
    fft_row = np.fft.fft(first_row)
    eig_sq = np.abs(fft_row) ** 2
    alpha = max(TRSD_ALPHA_REL * float(np.mean(eig_sq)), 1e-12)
    inv_denom = 1.0 / (eig_sq + alpha)

    x_real_raw = np.real(np.fft.ifft(np.conj(fft_row) * np.fft.fft(b_real) * inv_denom))
    x_imag_raw = np.real(np.fft.ifft(np.conj(fft_row) * np.fft.fft(b_imag) * inv_denom))
    z_r = np.clip(x_real_raw, 0.0, 1.0)
    z_i = np.clip(x_imag_raw, 0.0, 1.0)
    return np.concatenate([z_r, z_i])


def bch_decode_key(x_estimate: np.ndarray, n_blocks: int,
                   total_encoded: int) -> np.ndarray:
    """De-interleave and BCH decode to get K_group (256 bits)."""
    bch      = galois.BCH(BCH_N, BCH_K)
    x_bits   = (x_estimate[:total_encoded] >= 0.5).astype(int)
    # De-interleave
    deinterleaved = x_bits.reshape(BCH_N, n_blocks).T.flatten()
    decoded_blocks = []
    for b in range(n_blocks):
        cw   = galois.GF2(deinterleaved[b * BCH_N:(b + 1) * BCH_N])
        try:
            dec = np.array(bch.decode(cw), dtype=int)
        except Exception:
            dec = np.zeros(BCH_K, dtype=int)
        decoded_blocks.append(dec)
    return np.concatenate(decoded_blocks)  # 270 bits (9*30), take first 256


# ── Eve preparation ───────────────────────────────────────────────────────────

def load_eve_doss(exp: str, n_target: int) -> tuple[np.ndarray, np.ndarray]:
    """Load Eve's passive DOSS, BKRR-fill NaN, center, pad/trim to n_target."""
    base_exp = exp.replace("_eve", "")
    path     = EXP_DATA_DIR / base_exp / "doss_eve_alice.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Eve DOSS not found: {path}")

    df   = pd.read_csv(path)
    ab   = df["eve_approx_AB"].to_numpy(dtype=float)
    ac   = df["eve_approx_AC"].to_numpy(dtype=float)

    # BKRR fill
    ab = kalman_fill(ab)
    ac = kalman_fill(ac)

    # Smooth + center (mirror Stage 01)
    def _prep(s):
        if len(s) >= MA_WINDOW:
            s = pd.Series(s).rolling(MA_WINDOW, min_periods=1).mean().to_numpy()
        s = s - np.mean(s)
        return s

    ab, ac = _prep(ab), _prep(ac)

    # Align to n_target
    def _align(s):
        if len(s) >= n_target:
            return s[:n_target]
        return np.concatenate([s, np.zeros(n_target - len(s))])

    return _align(ab), _align(ac)


# ── Main analysis per experiment ──────────────────────────────────────────────

def run_eve_attack(exp: str) -> dict:
    base_exp = exp.replace("_eve", "")
    print(f"\n--- Eve Security Analysis: {base_exp} ---")

    # Load BCH params from base experiment
    bch_path = OUTPUT_DIR / f"03_BCH_params_{base_exp}.csv"
    if not bch_path.is_file():
        print(f"    [SKIP] BCH params not found ({bch_path.name}) — run base exp first")
        return {}
    bch_df     = pd.read_csv(bch_path).set_index("param")["value"].to_dict()
    n_ab       = int(bch_df["N_AB"])
    n_ac       = int(bch_df["N_AC"])
    n_blocks   = int(bch_df["L"])
    total_enc  = int(bch_df["total_encoded_bits"])
    key_bits   = int(bch_df["KEY_BITS"])

    # Load Alice's true K_group
    kg_path = OUTPUT_DIR / f"03_alice_K_group_{base_exp}.csv"
    if not kg_path.is_file():
        print(f"    [SKIP] K_group not found")
        return {}
    kg_df     = pd.read_csv(kg_path)
    k_real    = kg_df["real"].astype(int).to_numpy()
    k_imag    = kg_df["imag"].astype(int).to_numpy()
    k_group   = np.concatenate([k_real, k_imag])  # 256 bits

    # Load b_AB (Eve intercepts this — worst-case)
    b_ab_path = OUTPUT_DIR / f"03_alice_b_AB_{base_exp}.csv"
    b_ac_path = OUTPUT_DIR / f"03_alice_b_AC_{base_exp}.csv"
    if not b_ab_path.is_file():
        print(f"    [SKIP] b_AB not found")
        return {}
    b_ab_df  = pd.read_csv(b_ab_path)
    b_ab_r   = b_ab_df["Real"].to_numpy(dtype=float)
    b_ab_i   = b_ab_df["Imag"].to_numpy(dtype=float)
    b_ac_df  = pd.read_csv(b_ac_path)
    b_ac_r   = b_ac_df["Real"].to_numpy(dtype=float)
    b_ac_i   = b_ac_df["Imag"].to_numpy(dtype=float)

    # Load Alice's legitimate adjusted DOSS for correlation
    alice_adj_path = OUTPUT_DIR / f"01_alice_adjusted_AB_{base_exp}.csv"
    alice_adj = pd.read_csv(alice_adj_path)["adjusted_rssi"].to_numpy() if alice_adj_path.is_file() else np.array([])

    # ── Build Eve's matrices ──────────────────────────────────────────────────
    eve_ab, eve_ac = load_eve_doss(exp, n_target=max(n_ab, n_ac))
    eve_ab_n       = eve_ab[:n_ab]
    eve_ac_n       = eve_ac[:n_ac]

    # Pearson correlation: Alice vs Eve (AB link)
    if len(alice_adj) > 0:
        n_corr  = min(len(alice_adj), len(eve_ab_n))
        r_ae, _ = pearsonr(alice_adj[:n_corr], eve_ab_n[:n_corr])
    else:
        r_ae = float("nan")

    fr_eve_ab = build_first_row(eve_ab_n)
    fr_eve_ac = build_first_row(eve_ac_n)

    # ── Attack 1: Eve uses her H_AB to deobfuscate b_AB ──────────────────────
    print(f"    [EVE-AB] TRSD deobfuscation via Eve's H_AB (N={n_ab})")
    x_est_ab  = admm_deobfuscate(fr_eve_ab, b_ab_r, b_ab_i)
    key_eve_ab = bch_decode_key(x_est_ab, n_blocks, total_enc)[:key_bits]
    kdr_ab     = float(np.mean(key_eve_ab != k_group[:len(key_eve_ab)]))

    # ── Attack 2: Eve uses her H_AC to deobfuscate b_AC (relay path) ─────────
    print(f"    [EVE-AC] TRSD deobfuscation via Eve's H_AC (N={n_ac})")
    x_est_ac   = admm_deobfuscate(fr_eve_ac, b_ac_r, b_ac_i)
    key_eve_ac = bch_decode_key(x_est_ac, n_blocks, total_enc)[:key_bits]
    kdr_ac     = float(np.mean(key_eve_ac != k_group[:len(key_eve_ac)]))

    result = {
        "exp":          base_exp,
        "N_AB":         n_ab,
        "N_AC":         n_ac,
        "pearson_r_AliceEve_AB": round(r_ae, 4),
        "eve_kdr_via_AB":  f"{kdr_ab*100:.2f}%",
        "eve_kdr_via_AC":  f"{kdr_ac*100:.2f}%",
        "secure_AB":    "YES" if kdr_ab > 0.40 else "NO",
        "secure_AC":    "YES" if kdr_ac > 0.40 else "NO",
    }

    print(f"    [RESULT] r(Alice,Eve)_AB={r_ae:+.4f} | "
          f"Eve KDR via H_AB={kdr_ab*100:.2f}% | via H_AC={kdr_ac*100:.2f}%")
    print(f"    [SECURE] AB={'YES' if kdr_ab>0.40 else 'NO'} | "
          f"AC={'YES' if kdr_ac>0.40 else 'NO'}")
    return result


def run(exp: str) -> None:
    print(f"--- Stage 11: Eve Security Analysis ({exp}) ---")
    base_exp = exp.replace("_eve", "")

    result = run_eve_attack(exp)
    if not result:
        return

    out_path = OUTPUT_DIR / f"11_eve_results_{base_exp}.csv"
    pd.DataFrame([result]).to_csv(out_path, index=False)
    print(f"    [OK] {out_path.name} written")

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"  Eve Security Report — {base_exp}")
    print("=" * 65)
    for k, v in result.items():
        print(f"  {k:<35} {v}")
    print("=" * 65)
    print("Stage 11 complete.\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 11 — Eve security analysis (passive eavesdropping simulation)")
    parser.add_argument("--exp", required=True,
                        choices=["exp1", "exp2", "exp3",
                                 "exp1_eve", "exp2_eve", "exp3_eve"])
    args = parser.parse_args(argv)
    try:
        run(args.exp)
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
