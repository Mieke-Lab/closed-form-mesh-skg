"""
05_bob_trsd_deobfuscation.py
----------------------------
Stage 05 of the SKG pipeline (Bob side) — TRSD-based deobfuscation.

Solves the deobfuscation inverse problem
    minimize ||P_B x - b_AB||_2^2  s.t. x in [0,1]^N
using TRSD (Tikhonov-Regularized Spectral Decoding) — a one-shot
closed-form solver that exploits the circulant structure of P_B
via FFT diagonalization:

    x_hat = IFFT( conj(λ) · FFT(b) / (|λ|² + α) )
    x     = clip(x_hat, 0, 1)

where λ = FFT(first_row_of_P_B) are the eigenvalues and α is a
small Tikhonov regularization term. With CARR, the AC relay
constraint is used in place of the NLOS-corrupted H_BA.

After TRSD:
    1. Hard-threshold x at 0.5 to obtain a candidate bit vector x_hat.
    2. Mark uncertain indices where |x - 0.5| < confidence_threshold.
    3. Strip the zero-padding tail, de-interleave, and BCH(63,30,t=6) decode
       L = 9 blocks to recover the 256-bit K_group_B.

Outputs:
    output/05_bob_recovered_key_{exp}.csv    (real, imag bits)
    output/05_bob_error_indices_{exp}.json   (uncertain bit indices)
    output/05_bob_admm_report_{exp}.txt      (TRSD report — name kept for
                                              back-compat with downstream)
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

CONFIDENCE_TOL = 0.10   # |x - 0.5| < tol -> uncertain bit
MAX_FLIP_BITS = 5       # BCH trial-flip fallback budget per block

# TRSD hyper-parameters
TRSD_ALPHA_REL = 1e-3   # relative regularization (× mean eigenvalue energy)


def trsd_deobfuscate(first_row: np.ndarray,
                     b_real: np.ndarray,
                     b_imag: np.ndarray,
                     n_total: int,
                     alpha_rel: float = TRSD_ALPHA_REL,
                     first_row_ac: np.ndarray | None = None,
                     b_ac_real: np.ndarray | None = None,
                     b_ac_imag: np.ndarray | None = None,
                     **_unused) -> tuple[np.ndarray, dict]:
    """Tikhonov-Regularized Spectral Decoding (TRSD).

    One-shot closed-form deobfuscation for circulant systems via FFT
    diagonalization. Replaces iterative ADMM with a single FFT-based
    regularized inverse:

        x̂ = IFFT( conj(λ) · FFT(b) / (|λ|² + α) )

    where λ = FFT(first_row) are the eigenvalues of the circulant matrix
    and α is a small Tikhonov regularization coefficient. The result is
    then clipped to [0,1] to enforce the binary-projection constraint.

    With CARR: uses H_AC eigenvalues, bypasses corrupted H_BA (NLOS).
    """
    fft_row = np.fft.fft(first_row)
    eig_sq = np.abs(fft_row) ** 2

    use_carr = (first_row_ac is not None and b_ac_real is not None)
    if use_carr:
        n_main = len(b_real)

        def _align(arr):
            if len(arr) >= n_main:
                return arr[:n_main]
            return np.concatenate([arr, np.zeros(n_main - len(arr))])

        fr_ac = _align(first_row_ac.astype(float))
        b_ac_r = _align(b_ac_real.astype(float))
        b_ac_i = _align(b_ac_imag.astype(float))
        fft_row_use = np.fft.fft(fr_ac)
        eig_sq_use = np.abs(fft_row_use) ** 2
        b_real_use = b_ac_r
        b_imag_use = b_ac_i
        print("    [CARR] AC-only TRSD: relay constraint active (H_BA NLOS bypassed)")
    else:
        fft_row_use = fft_row
        eig_sq_use = eig_sq
        b_real_use = b_real
        b_imag_use = b_imag

    # Adaptive regularization: scale by mean eigenvalue energy
    alpha = alpha_rel * float(np.mean(eig_sq_use))
    if alpha < 1e-12:
        alpha = 1e-12
    inv_denom = 1.0 / (eig_sq_use + alpha)

    # TRSD one-shot inversion (real and imag branches)
    x_real_raw = np.real(np.fft.ifft(np.conj(fft_row_use) * np.fft.fft(b_real_use) * inv_denom))
    x_imag_raw = np.real(np.fft.ifft(np.conj(fft_row_use) * np.fft.fft(b_imag_use) * inv_denom))

    # Box projection onto [0,1]
    z_real = np.clip(x_real_raw, 0.0, 1.0)
    z_imag = np.clip(x_imag_raw, 0.0, 1.0)

    # Reconstruction residual (how far raw solution from box)
    primal_real = float(np.linalg.norm(z_real - x_real_raw))
    primal_imag = float(np.linalg.norm(z_imag - x_imag_raw))

    x_full = np.concatenate([z_real, z_imag])
    info = {
        "real_residual": primal_real,
        "imag_residual": primal_imag,
        "real_iterations": 1,
        "imag_iterations": 1,
        "rho_final_real": alpha,
        "rho_final_imag": alpha,
        "alpha": alpha,
    }
    return x_full[:n_total], info


def deinterleave(bits: np.ndarray, n_blocks: int, bch_n: int) -> np.ndarray:
    """Inverse of Alice's row-write column-read interleaver."""
    return bits.reshape(bch_n, n_blocks).T.flatten()


def _decode_block_verified(bch_codec, block_bits: np.ndarray,
                           soft_vals: np.ndarray, bch_t: int, bch_k: int):
    """Decode one BCH block with re-encode verification + trial-flip fallback.

    1. Direct galois decode; verify by re-encoding and checking Hamming
       distance <= t.
    2. If verification fails, flip the most-uncertain bits (ordered by
       |soft - 0.5|) up to MAX_FLIP_BITS at a time and retry.
    """
    block_gf = galois.GF2(block_bits)
    try:
        decoded_msg = bch_codec.decode(block_gf)
        re_encoded = bch_codec.encode(decoded_msg)
        hd = int(np.sum(np.array(re_encoded, dtype=int)
                        != np.array(block_gf, dtype=int)))
        if hd <= bch_t:
            return np.array(decoded_msg[:bch_k], dtype=int), "direct"
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
                hd = int(np.sum(np.array(re_encoded, dtype=int)
                                != np.array(trial_gf, dtype=int)))
                if hd <= bch_t:
                    return np.array(decoded_msg[:bch_k], dtype=int), "trial-flip"
            except Exception:
                continue
    return None, None


def bch_decode(interleaved_estimate: np.ndarray,
               n_blocks: int, bch_n: int, bch_k: int, key_bits: int,
               interleaved_soft: np.ndarray | None = None,
               bch_t: int = 6) -> np.ndarray:
    """Decode L blocks of BCH(N,K,t) with trial-flip fallback.

    When `interleaved_soft` is supplied, each block is decoded via
    `_decode_block_verified`, which falls back to flipping up to
    MAX_FLIP_BITS least-confident bits when the native galois decoder
    fails verification.
    """
    bch = galois.BCH(bch_n, bch_k)
    de_il = deinterleave(interleaved_estimate, n_blocks, bch_n)
    soft_de_il = (deinterleave(interleaved_soft, n_blocks, bch_n)
                  if interleaved_soft is not None else None)
    decoded = []
    flip_recoveries = 0
    decode_failures = 0
    for b in range(n_blocks):
        codeword = de_il[b * bch_n:(b + 1) * bch_n].astype(int)
        if soft_de_il is not None:
            soft_block = soft_de_il[b * bch_n:(b + 1) * bch_n]
            msg, method = _decode_block_verified(bch, codeword, soft_block,
                                                 bch_t, bch_k)
            if msg is None:
                decode_failures += 1
                msg = codeword[:bch_k]
            elif method == "trial-flip":
                flip_recoveries += 1
            decoded.append(msg)
        else:
            msg = bch.decode(galois.GF2(codeword))
            decoded.append(np.array(msg, dtype=int))
    if flip_recoveries:
        print(f"    [BCH] {flip_recoveries} block(s) recovered via trial-flip")
    if decode_failures:
        print(f"    [BCH] {decode_failures} block(s) failed BCH decode")
    all_msg = np.concatenate(decoded)
    return all_msg[:key_bits]


def run(exp: str) -> None:
    print(f"--- Stage 05: Bob TRSD Deobfuscation ({exp}) ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    b_path = OUTPUT_DIR / f"03_alice_b_AB_{exp}.csv"
    mat_path = OUTPUT_DIR / f"04_bob_matrix_BA_{exp}.csv"
    bch_path = OUTPUT_DIR / f"03_BCH_params_{exp}.csv"
    for p in (b_path, mat_path, bch_path):
        if not p.is_file():
            raise FileNotFoundError(f"Required input missing: {p}")

    b_df = pd.read_csv(b_path)
    b_real = b_df["Real"].to_numpy(dtype=float)
    b_imag = b_df["Imag"].to_numpy(dtype=float)

    first_row = pd.read_csv(mat_path)["first_row"].to_numpy(dtype=float)

    # CARR: load Alice-Charlie relay constraint if available
    b_ac_path = OUTPUT_DIR / f"03_alice_b_AC_{exp}.csv"
    mat_ac_path = OUTPUT_DIR / f"02_alice_matrix_AC_{exp}.csv"
    first_row_ac = b_ac_real = b_ac_imag = None
    if b_ac_path.is_file() and mat_ac_path.is_file():
        b_ac_df = pd.read_csv(b_ac_path)
        b_ac_real = b_ac_df["Real"].to_numpy(dtype=float)
        b_ac_imag = b_ac_df["Imag"].to_numpy(dtype=float)
        first_row_ac = pd.read_csv(mat_ac_path)["first_row"].to_numpy(dtype=float)
        print(f"    [CARR] relay files loaded: b_AC N={len(b_ac_real)}, matrix_AC N={len(first_row_ac)}")
    else:
        print("    [CARR] relay files not found — single-constraint TRSD only")

    bch_df = pd.read_csv(bch_path)
    params = dict(zip(bch_df["param"].astype(str), bch_df["value"]))
    bch_n = int(params["BCH_N"])
    bch_k = int(params["BCH_K"])
    bch_t = int(params["BCH_T"])
    n_blocks = int(params["L"])
    key_bits = int(params["KEY_BITS"])
    total_encoded = int(params["total_encoded_bits"])
    n_alice = int(params["N_AB"])

    if first_row.shape[0] != b_real.shape[0]:
        print(f"    [WARN] dim mismatch P_B (N={first_row.shape[0]}) vs b_AB "
              f"(N={b_real.shape[0]}). TRSD will use N from b_AB.")
        # Pad/trim first_row to match b length to keep FFT consistent.
        n_b = b_real.shape[0]
        if first_row.shape[0] < n_b:
            first_row = np.concatenate([first_row, np.zeros(n_b - first_row.shape[0])])
        else:
            first_row = first_row[:n_b]

    n_real = b_real.shape[0]
    n_full = 2 * n_real

    print(f"    [TRSD] alpha_rel={TRSD_ALPHA_REL}, N(real)={n_real}")
    t0 = time.perf_counter()
    x_estimate, info = trsd_deobfuscate(
        first_row, b_real, b_imag, n_total=n_full,
        first_row_ac=first_row_ac, b_ac_real=b_ac_real, b_ac_imag=b_ac_imag)
    elapsed = time.perf_counter() - t0
    print(f"    [TRSD] residuals real={info['real_residual']:.4e} "
          f"imag={info['imag_residual']:.4e} alpha={info['alpha']:.4e} "
          f"wall={elapsed*1000:.2f}ms")

    # Trim padded portion -> interleaved estimate
    if total_encoded > x_estimate.size:
        raise ValueError(f"total_encoded ({total_encoded}) > x_estimate size ({x_estimate.size})")
    interleaved_estimate_soft = x_estimate[:total_encoded]
    interleaved_estimate = (interleaved_estimate_soft >= 0.5).astype(int)

    # Data-driven confidence threshold:
    # sigma = projection_residual / sqrt(N) — normalised noise estimate.
    # confidence_tol = clip(3*sigma, 0.05, 0.20)
    sigma_final = (info["real_residual"] + info["imag_residual"]) / (2.0 * np.sqrt(n_real))
    confidence_tol = float(np.clip(sigma_final * 3.0, 0.05, 0.20))
    distance = np.abs(interleaved_estimate_soft - 0.5)
    uncertain_mask = distance < confidence_tol
    uncertain_indices = np.where(uncertain_mask)[0].tolist()
    print(f"    [TRSD] sigma={sigma_final:.4e} conf_tol={confidence_tol:.3f} "
          f"alpha={info['alpha']:.4e}")

    # Persist interleaved hard estimate so Stage 06 can patch without re-encoding
    il_path = OUTPUT_DIR / f"05_bob_interleaved_estimate_{exp}.csv"
    pd.DataFrame({"bit": interleaved_estimate}).to_csv(il_path, index=False)

    # BCH decode to recover K_group_B
    k_group_b = bch_decode(interleaved_estimate, n_blocks, bch_n, bch_k, key_bits,
                           interleaved_soft=interleaved_estimate_soft,
                           bch_t=bch_t)
    real_bits = k_group_b[: key_bits // 2].astype(int)
    imag_bits = k_group_b[key_bits // 2:].astype(int)

    rec_path = OUTPUT_DIR / f"05_bob_recovered_key_{exp}.csv"
    pd.DataFrame({"real": real_bits, "imag": imag_bits}).to_csv(rec_path, index=False)

    err_path = OUTPUT_DIR / f"05_bob_error_indices_{exp}.json"
    with err_path.open("w") as fh:
        json.dump(
            {
                "exp": exp,
                "total_bits": int(total_encoded),
                "uncertain_count": int(len(uncertain_indices)),
                "confidence_tol": CONFIDENCE_TOL,
                "indices": uncertain_indices,
            },
            fh,
            indent=2,
        )

    rep_path = OUTPUT_DIR / f"05_bob_admm_report_{exp}.txt"
    with rep_path.open("w") as fh:
        fh.write(f"exp={exp}\n")
        fh.write(f"N={n_real}\n")
        fh.write(f"solver=TRSD\n")
        fh.write(f"alpha_rel={TRSD_ALPHA_REL}\n")
        fh.write(f"alpha={info['alpha']:.6e}\n")
        fh.write(f"real_iterations={info['real_iterations']}\n")
        fh.write(f"imag_iterations={info['imag_iterations']}\n")
        fh.write(f"real_residual={info['real_residual']:.6e}\n")
        fh.write(f"imag_residual={info['imag_residual']:.6e}\n")
        fh.write(f"uncertain_bits={len(uncertain_indices)}/{total_encoded}\n")
        fh.write(f"confidence_tol={confidence_tol:.4f}\n")
        fh.write(f"BCH=({bch_n},{bch_k},t={bch_t}) blocks={n_blocks}\n")
        fh.write(f"wall_seconds={elapsed:.3f}\n")
    print(f"    [OK] {rec_path.name}, {il_path.name}, {err_path.name}, {rep_path.name} written")
    print(f"    [SUMMARY] uncertain_bits={len(uncertain_indices)}/{total_encoded}")
    print("Stage 05 complete.\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bob Stage 05 - TRSD deobfuscation")
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
