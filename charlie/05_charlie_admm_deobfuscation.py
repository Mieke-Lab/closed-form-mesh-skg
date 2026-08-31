"""
05_charlie_trsd_deobfuscation.py
--------------------------------
Stage 05 of the SKG pipeline (Charlie side) — TRSD-based deobfuscation.

Mirror of Bob's deobfuscation but for channel AC. Charlie solves the
inverse problem
    minimize ||P_C * x_tilde - b_AC||^2  s.t. x in [0,1]^{2N}
using TRSD (Tikhonov-Regularized Spectral Decoding) — a one-shot
closed-form solver that exploits the circulant structure of P_C via
FFT diagonalization:

    x_hat = IFFT( conj(λ) · FFT(b) / (|λ|² + α) )
    x     = clip(x_hat, 0, 1)

After TRSD, Charlie:
  1. extracts binary key candidate via threshold 0.5,
  2. records uncertain positions (|x - 0.5| < epsilon),
  3. de-interleaves and BCH(63,30,t=6) decodes 9 blocks (with trial-flip
     recovery using soft confidence),
  4. recovers K_group_C (256-bit candidate).

Outputs:
    output/05_charlie_estimated_key_{exp}.csv  (real/imag, soft+rounded)
    output/05_charlie_recovered_key_{exp}.csv  (Real,Imag bits)
    output/05_charlie_error_indices_{exp}.json (uncertain bit indices)
    output/05_charlie_admm_report_{exp}.txt    (TRSD report; filename kept
                                                for downstream back-compat)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import galois
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

TRSD_ALPHA_REL = 1e-3  # relative Tikhonov regularization
EPSILON_CONFIDENCE = 0.05  # uncertain bit threshold (matches Bob)
MAX_FLIP_BITS = 5


def try_decode_verified(bch_codec: galois.BCH, block_bits: np.ndarray,
                        soft_vals: np.ndarray, t: int, k: int):
    """Decode BCH block and verify Hamming distance <= t. Trial-flip if needed."""
    block_gf = galois.GF2(block_bits)
    try:
        decoded_msg = bch_codec.decode(block_gf)
        re_encoded = bch_codec.encode(decoded_msg)
        hd = int(np.sum(np.array(re_encoded, dtype=int) != np.array(block_gf, dtype=int)))
        if hd <= t:
            return np.array(decoded_msg[:k], dtype=int), "direct"
    except Exception:
        pass

    uncertainty = np.abs(soft_vals - 0.5)
    order = np.argsort(uncertainty)
    candidates = min(MAX_FLIP_BITS * 3, len(block_bits))
    for num_flips in range(1, MAX_FLIP_BITS + 1):
        for combo in combinations(range(candidates), num_flips):
            trial = block_bits.copy()
            for ci in combo:
                trial[order[ci]] ^= 1
            trial_gf = galois.GF2(trial)
            try:
                decoded_msg = bch_codec.decode(trial_gf)
                re_encoded = bch_codec.encode(decoded_msg)
                hd = int(np.sum(np.array(re_encoded, dtype=int) != np.array(trial_gf, dtype=int)))
                if hd <= t:
                    return np.array(decoded_msg[:k], dtype=int), "trial-flip"
            except Exception:
                continue
    return None, None


def run(exp: str) -> None:
    print(f"--- Stage 05: Charlie TRSD Deobfuscation ({exp}) ---")
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Inputs.
    matrix_path = OUTPUT_DIR / f"04_charlie_matrix_AC_{exp}.csv"
    package_path = OUTPUT_DIR / f"03_alice_b_AC_{exp}.csv"
    bch_path = OUTPUT_DIR / f"03_BCH_params_{exp}.csv"
    for p in (matrix_path, package_path, bch_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing input: {p}")

    print("1/6 Reading inputs (matrix, package, BCH params)...")
    first_row_C = pd.read_csv(matrix_path)["first_row"].to_numpy(dtype=float)
    N = len(first_row_C)

    df_b = pd.read_csv(package_path)
    b_real = df_b["Real"].to_numpy(dtype=float)
    b_imag = df_b["Imag"].to_numpy(dtype=float)
    if len(b_real) != N or len(b_imag) != N:
        raise ValueError(f"b_AC length {len(b_real)} != matrix N {N}")

    bch_dict = dict(zip(
        pd.read_csv(bch_path)["param"],
        pd.read_csv(bch_path)["value"].astype(int)
    ))
    BCH_N = bch_dict["BCH_N"]
    BCH_K = bch_dict["BCH_K"]
    BCH_T = bch_dict["BCH_T"]
    L = bch_dict["L"]
    KEY_BITS = bch_dict["KEY_BITS"]
    total_encoded_bits = bch_dict["total_encoded_bits"]
    print(f"    [OK] N={N}, BCH({BCH_N},{BCH_K},t={BCH_T}), L={L}, "
          f"key={KEY_BITS} bit, encoded={total_encoded_bits} bit")

    # 2. FFT setup for circulant operators.
    print("2/6 FFT setup for TRSD...")
    eig_C = np.fft.fft(first_row_C)
    eig_PtP = np.abs(eig_C) ** 2

    # TRSD: adaptive Tikhonov regularization (× mean eigenvalue energy)
    alpha = max(TRSD_ALPHA_REL * float(np.mean(eig_PtP)), 1e-12)
    inv_denom = 1.0 / (eig_PtP + alpha)
    print(f"    [INFO] alpha_rel={TRSD_ALPHA_REL}, alpha={alpha:.4e}")

    # 3. TRSD one-shot inversion (closed-form, no iteration).
    print("3/6 Running TRSD (one-shot closed-form)...")
    t_trsd_start = time.time()
    x_real_raw = np.real(np.fft.ifft(np.conj(eig_C) * np.fft.fft(b_real) * inv_denom))
    x_imag_raw = np.real(np.fft.ifft(np.conj(eig_C) * np.fft.fft(b_imag) * inv_denom))

    # Box projection onto [0,1]
    y_real = np.clip(x_real_raw, 0.0, 1.0)
    y_imag = np.clip(x_imag_raw, 0.0, 1.0)
    t_trsd_elapsed = time.time() - t_trsd_start

    final_iter = 1
    final_residual = float(np.sqrt(
        np.sum((x_real_raw - y_real) ** 2) + np.sum((x_imag_raw - y_imag) ** 2)))
    print(f"    [OK] TRSD complete in {t_trsd_elapsed*1000:.2f}ms, "
          f"projection_residual={final_residual:.6e}")

    # 4. Round to bits, persist soft + hard estimate.
    print("4/6 Saving estimated key (soft + rounded)...")
    real_rounded = np.clip(np.round(y_real).astype(int), 0, 1)
    imag_rounded = np.clip(np.round(y_imag).astype(int), 0, 1)
    pd.DataFrame({
        "Real_Est": y_real,
        "Imag_Est": y_imag,
        "Real_Rounded": real_rounded,
        "Imag_Rounded": imag_rounded,
    }).to_csv(OUTPUT_DIR / f"05_charlie_estimated_key_{exp}.csv", index=False)

    # 5. Identify uncertain bits in info region.
    print("5/6 Identifying uncertain bits (|x-0.5|<epsilon)...")
    all_est = np.concatenate([y_real, y_imag])
    info_est = all_est[:total_encoded_bits]
    error_indices = [int(i) for i in range(total_encoded_bits)
                     if abs(info_est[i] - 0.5) < EPSILON_CONFIDENCE]
    err_path = OUTPUT_DIR / f"05_charlie_error_indices_{exp}.json"
    with err_path.open("w") as fh:
        json.dump(error_indices, fh)
    print(f"    [OK] {len(error_indices)}/{total_encoded_bits} uncertain bits "
          f"-> {err_path.name}")

    # 6. BCH decode the info region with trial-flip recovery.
    print("6/6 BCH decoding...")
    bch = galois.BCH(BCH_N, BCH_K)
    all_rounded = np.concatenate([real_rounded, imag_rounded])
    x_info = all_rounded[:total_encoded_bits]
    x_deint = x_info.reshape(BCH_N, L).T.flatten()
    soft_info = info_est
    soft_deint = soft_info.reshape(BCH_N, L).T.flatten()

    decoded_blocks = []
    failures = 0
    trial_flip_recoveries = 0
    for i in range(L):
        block_bits = x_deint[i * BCH_N:(i + 1) * BCH_N].copy()
        block_soft = soft_deint[i * BCH_N:(i + 1) * BCH_N]
        msg, method = try_decode_verified(bch, block_bits, block_soft, BCH_T, BCH_K)
        if msg is None:
            failures += 1
            decoded_blocks.append(np.array(block_bits[:BCH_K], dtype=int))
        else:
            decoded_blocks.append(msg)
            if method == "trial-flip":
                trial_flip_recoveries += 1

    K_padded = np.concatenate(decoded_blocks)
    K_group = K_padded[:KEY_BITS]
    half = KEY_BITS // 2
    K_real = K_group[:half]
    K_imag = K_group[half:]

    pd.DataFrame({"Real": K_real, "Imag": K_imag}).to_csv(
        OUTPUT_DIR / f"05_charlie_recovered_key_{exp}.csv", index=False
    )

    elapsed = time.time() - t_start
    report_path = OUTPUT_DIR / f"05_charlie_admm_report_{exp}.txt"
    with report_path.open("w") as fh:
        fh.write("# Stage 05 (Charlie) - TRSD deobfuscation report\n")
        fh.write(f"exp                = {exp}\n")
        fh.write(f"N                  = {N}\n")
        fh.write(f"solver             = TRSD\n")
        fh.write(f"alpha_rel          = {TRSD_ALPHA_REL}\n")
        fh.write(f"alpha              = {alpha:.6e}\n")
        fh.write(f"iterations         = {final_iter}\n")
        fh.write(f"projection_residual= {final_residual:.6e}\n")
        fh.write(f"epsilon_confidence = {EPSILON_CONFIDENCE}\n")
        fh.write(f"uncertain_bits     = {len(error_indices)}\n")
        fh.write(f"trial_flip_recov   = {trial_flip_recoveries}\n")
        fh.write(f"bch_decode_fail    = {failures}\n")
        fh.write(f"elapsed_seconds    = {elapsed:.3f}\n")
    print(f"    [OK] report -> {report_path.name} (elapsed {elapsed:.2f}s)")
    print("Stage 05 complete.\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Charlie Stage 05 - TRSD deobfuscation")
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
