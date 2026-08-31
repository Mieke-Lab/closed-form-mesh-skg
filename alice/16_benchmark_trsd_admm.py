"""
16_benchmark_trsd_admm.py
-------------------------
Empirical head-to-head benchmark: TRSD (closed-form Tikhonov spectral
decode) vs ADMM (iterative consensus solver per [Du et al. 2024]).

Per-link decode benchmark: both solvers operate on the *same* per-link
input (H_BA, b_AB) at Bob's stage 05 (deobfuscation). Multi-user
orchestration (DOSS triangle, CARR, reconciliation) is unchanged across
both choices — TRSD/ADMM is a drop-in decode-step substitution.

Inputs per scenario:
    output/04_bob_matrix_BA_{exp}.csv   — circulant first row (pulled
                                          from Bob via SSH)
    output/03_alice_b_AB_{exp}.csv     — Alice's encapsulated parities
    output/03_BCH_params_{exp}.csv     — BCH(N, K, t) parameters

Methods compared:
    1. TRSD : x̂ = IFFT(λ̄ ⊙ FFT(b) / (|λ|² + λ_reg))
               λ_reg = max(1e-3 · mean|λ|², 1e-12)
               Single FFT/IFFT pass.
    2. ADMM : FFT-accelerated ADMM per Du et al. 2024
              max_iter = 200, tol = 1e-4 on primal residual.
              Auto-scaled rho = ‖P_T b‖ / √(2N), clamped ≥ 0.1.
              Box projection y ∈ [0,1]^N.

Metrics:
    - Wall-time (mean over R runs, default R=10)
    - Iterations to converge (ADMM only)
    - Final reconstruction residual ‖z - x̂_raw‖
    - Uncertain-bit count (hard-decision ε-band, ε = 0.05)
    - Bit-mismatch vs TRSD output (sanity check; should be ≤ 1-2 bits)

Output:
    output/16_benchmark_{exp}.txt   — per-exp report
    output/16_benchmark_summary.csv — aggregate table

Usage:
    python3 16_benchmark_trsd_admm.py             # all exps, R=10 runs
    python3 16_benchmark_trsd_admm.py --runs 5    # fewer reps
    python3 16_benchmark_trsd_admm.py --exp exp1  # single exp
"""
from __future__ import annotations
import os

import argparse
import subprocess
import sys
import tempfile
import time
from itertools import combinations
from pathlib import Path

import galois
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

BOB_HOST = os.environ.get("BOB_HOST", "bob@10.0.0.2")
BOB_PASS = os.environ.get("BOB_PASS", "")
BOB_REMOTE = ("/home/bob/secret-key-generation-collection/magisterSeries/"
              "2026/IES_basedPTM_No3/channelProb/output")

EPSILON = 0.05          # uncertain-bit band
ADMM_MAX_ITER = 200
ADMM_TOL = 1e-4


def scp_pull(remote_name: str, local_path: Path) -> bool:
    cmd = ["sshpass", "-p", BOB_PASS, "scp",
           "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
           f"{BOB_HOST}:{BOB_REMOTE}/{remote_name}", str(local_path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"  [WARN] scp {remote_name} failed: "
              f"{exc.stderr.decode(errors='ignore').strip()}")
        return False


def trsd_decode(first_row: np.ndarray,
                b_real: np.ndarray, b_imag: np.ndarray,
                alpha_rel: float = 1e-3) -> tuple[np.ndarray, np.ndarray, dict]:
    """One-shot Tikhonov-regularised circulant inverse via FFT."""
    fft_row = np.fft.fft(first_row)
    eig_sq = np.abs(fft_row) ** 2
    alpha = max(alpha_rel * float(np.mean(eig_sq)), 1e-12)
    inv_denom = 1.0 / (eig_sq + alpha)
    x_real = np.real(np.fft.ifft(np.conj(fft_row) * np.fft.fft(b_real) * inv_denom))
    x_imag = np.real(np.fft.ifft(np.conj(fft_row) * np.fft.fft(b_imag) * inv_denom))
    z_real = np.clip(x_real, 0.0, 1.0)
    z_imag = np.clip(x_imag, 0.0, 1.0)
    primal = float(np.linalg.norm(z_real - x_real) + np.linalg.norm(z_imag - x_imag))
    return z_real, z_imag, {"alpha": alpha, "residual": primal, "iters": 1}


def admm_decode(first_row: np.ndarray,
                b_real: np.ndarray, b_imag: np.ndarray,
                max_iter: int = ADMM_MAX_ITER,
                tol: float = ADMM_TOL) -> tuple[np.ndarray, np.ndarray, dict]:
    """FFT-accelerated ADMM per Du 2024 KeyCome: x = argmin ||Hx - b||² s.t. x∈[0,1]."""
    N = len(first_row)
    fft_row = np.fft.fft(first_row)
    eig_PtP = np.abs(fft_row) ** 2

    Pt_b_real = np.real(np.fft.ifft(np.conj(fft_row) * np.fft.fft(b_real)))
    Pt_b_imag = np.real(np.fft.ifft(np.conj(fft_row) * np.fft.fft(b_imag)))

    Pt_b_norm = np.sqrt(np.sum(Pt_b_real ** 2) + np.sum(Pt_b_imag ** 2))
    rho = max(Pt_b_norm / np.sqrt(2 * N), 0.1)
    inv_eig = 1.0 / (eig_PtP + rho)

    x_r = np.zeros(N); y_r = np.zeros(N); lam_r = np.zeros(N)
    x_i = np.zeros(N); y_i = np.zeros(N); lam_i = np.zeros(N)

    converged_at = max_iter
    for k in range(max_iter):
        rhs_r = Pt_b_real + rho * y_r - lam_r
        rhs_i = Pt_b_imag + rho * y_i - lam_i
        x_r = np.real(np.fft.ifft(inv_eig * np.fft.fft(rhs_r)))
        x_i = np.real(np.fft.ifft(inv_eig * np.fft.fft(rhs_i)))
        y_r = np.clip(x_r + lam_r / rho, 0.0, 1.0)
        y_i = np.clip(x_i + lam_i / rho, 0.0, 1.0)
        lam_r += rho * (x_r - y_r)
        lam_i += rho * (x_i - y_i)
        if (k + 1) % 10 == 0:
            primal = np.sqrt(np.sum((x_r - y_r) ** 2) + np.sum((x_i - y_i) ** 2))
            if primal < tol:
                converged_at = k + 1
                break

    primal = float(np.linalg.norm(x_r - y_r) + np.linalg.norm(x_i - y_i))
    return y_r, y_i, {"rho": rho, "residual": primal, "iters": converged_at}


def hard_decision(z: np.ndarray) -> np.ndarray:
    return (z >= 0.5).astype(int)


def uncertain_count(z: np.ndarray, eps: float = EPSILON) -> int:
    return int(np.sum(np.abs(z - 0.5) < eps))


MAX_FLIP_BITS = 5  # BCH trial-flip fallback budget per block


def deinterleave(bits: np.ndarray, n_blocks: int, bch_n: int) -> np.ndarray:
    return bits.reshape(bch_n, n_blocks).T.flatten()


def _decode_block_verified(bch_codec, block_bits, soft_vals, bch_t, bch_k):
    """BCH decode one block with re-encode verification + trial-flip fallback."""
    block_gf = galois.GF2(block_bits)
    try:
        msg = bch_codec.decode(block_gf)
        re_enc = bch_codec.encode(msg)
        hd = int(np.sum(np.array(re_enc, dtype=int) != np.array(block_gf, dtype=int)))
        if hd <= bch_t:
            return np.array(msg[:bch_k], dtype=int)
    except Exception:
        pass
    order = np.argsort(np.abs(soft_vals - 0.5))
    cand = min(MAX_FLIP_BITS * 3, len(block_bits))
    for nf in range(1, MAX_FLIP_BITS + 1):
        for combo in combinations(range(cand), nf):
            trial = block_bits.copy()
            for ci in combo:
                trial[order[ci]] ^= 1
            tg = galois.GF2(trial)
            try:
                msg = bch_codec.decode(tg)
                re_enc = bch_codec.encode(msg)
                hd = int(np.sum(np.array(re_enc, dtype=int) != np.array(tg, dtype=int)))
                if hd <= bch_t:
                    return np.array(msg[:bch_k], dtype=int)
            except Exception:
                continue
    return block_bits[:bch_k]


def bch_decode_to_key(zr: np.ndarray, zi: np.ndarray,
                     bch_n: int, bch_k: int, bch_t: int,
                     n_blocks: int, key_bits: int,
                     total_encoded: int) -> np.ndarray:
    """Hard-decision → deinterleave → BCH decode (per block) → key_bits."""
    soft = np.concatenate([zr, zi])[:total_encoded]
    hard = (soft >= 0.5).astype(int)
    de = deinterleave(hard, n_blocks, bch_n)
    de_soft = deinterleave(soft, n_blocks, bch_n)
    bch = galois.BCH(bch_n, bch_k)
    out = []
    for b in range(n_blocks):
        cw = de[b * bch_n:(b + 1) * bch_n]
        sv = de_soft[b * bch_n:(b + 1) * bch_n]
        msg = _decode_block_verified(bch, cw, sv, bch_t, bch_k)
        out.append(msg)
    return np.concatenate(out)[:key_bits]


def benchmark_exp(exp: str, runs: int = 10) -> dict:
    print(f"\n=== Benchmark {exp} ===")

    # Fetch inputs
    tmp = Path(tempfile.mkdtemp(prefix=f"bench_{exp}_"))
    mat_remote = f"04_bob_matrix_BA_{exp}.csv"
    mat_local = tmp / mat_remote
    if not scp_pull(mat_remote, mat_local):
        return {"exp": exp, "error": "scp_failed"}

    b_path = OUTPUT_DIR / f"03_alice_b_AB_{exp}.csv"
    if not b_path.is_file():
        print(f"  [ERROR] missing {b_path}")
        return {"exp": exp, "error": "missing_b_AB"}

    first_row = pd.read_csv(mat_local)["first_row"].to_numpy(dtype=float)
    b_df = pd.read_csv(b_path)
    b_real = b_df["Real"].to_numpy(dtype=float)
    b_imag = b_df["Imag"].to_numpy(dtype=float)

    n_main = len(b_real)
    if first_row.shape[0] != n_main:
        if first_row.shape[0] < n_main:
            first_row = np.concatenate([first_row, np.zeros(n_main - first_row.shape[0])])
        else:
            first_row = first_row[:n_main]

    print(f"  N={n_main}, runs={runs}")

    # TRSD timing (R runs)
    trsd_times = []
    trsd_info = None
    for _ in range(runs):
        t0 = time.perf_counter()
        zr, zi, info = trsd_decode(first_row, b_real, b_imag)
        trsd_times.append(time.perf_counter() - t0)
        trsd_info = info

    # ADMM timing (R runs)
    admm_times = []
    admm_info = None
    for _ in range(runs):
        t0 = time.perf_counter()
        yr, yi, info = admm_decode(first_row, b_real, b_imag)
        admm_times.append(time.perf_counter() - t0)
        admm_info = info

    # Sanity: bit comparison TRSD vs ADMM at codeword level
    zr_t, zi_t, _ = trsd_decode(first_row, b_real, b_imag)
    yr_a, yi_a, _ = admm_decode(first_row, b_real, b_imag)
    bits_trsd = np.concatenate([hard_decision(zr_t), hard_decision(zi_t)])
    bits_admm = np.concatenate([hard_decision(yr_a), hard_decision(yi_a)])
    bit_diff = int(np.sum(bits_trsd != bits_admm))

    unc_trsd = uncertain_count(zr_t) + uncertain_count(zi_t)
    unc_admm = uncertain_count(yr_a) + uncertain_count(yi_a)

    # K_group-level comparison (end-to-end via BCH decode)
    kg_truth_path = OUTPUT_DIR / f"03_alice_K_group_{exp}.csv"
    bch_params_path = OUTPUT_DIR / f"03_BCH_params_{exp}.csv"
    kdr_trsd = kdr_admm = -1
    try:
        kg_df = pd.read_csv(kg_truth_path)
        kg_df.columns = [c.lower() for c in kg_df.columns]
        kg_real = kg_df["real"].to_numpy(int)
        kg_imag = kg_df["imag"].to_numpy(int)
        kg_truth = np.concatenate([kg_real, kg_imag])
        bp = pd.read_csv(bch_params_path)
        bp_map = dict(zip(bp["param"].astype(str), bp["value"]))
        bch_n = int(bp_map["BCH_N"]); bch_k = int(bp_map["BCH_K"])
        bch_t = int(bp_map["BCH_T"]); n_blocks = int(bp_map["L"])
        key_bits = int(bp_map["KEY_BITS"])
        total_enc = int(bp_map["total_encoded_bits"])
        k_trsd = bch_decode_to_key(zr_t, zi_t, bch_n, bch_k, bch_t,
                                    n_blocks, key_bits, total_enc)
        k_admm = bch_decode_to_key(yr_a, yi_a, bch_n, bch_k, bch_t,
                                    n_blocks, key_bits, total_enc)
        kdr_trsd = int(np.sum(k_trsd != kg_truth))
        kdr_admm = int(np.sum(k_admm != kg_truth))
    except Exception as exc:
        print(f"  [WARN] K_group cmp skipped: {exc}")

    trsd_arr = np.array(trsd_times) * 1000  # ms
    admm_arr = np.array(admm_times) * 1000

    result = {
        "exp": exp,
        "N": n_main,
        "runs": runs,
        "trsd_mean_ms": float(np.mean(trsd_arr)),
        "trsd_std_ms": float(np.std(trsd_arr, ddof=1)) if runs > 1 else 0.0,
        "trsd_min_ms": float(np.min(trsd_arr)),
        "trsd_iters": trsd_info["iters"],
        "trsd_residual": trsd_info["residual"],
        "trsd_uncertain_bits": unc_trsd,
        "admm_mean_ms": float(np.mean(admm_arr)),
        "admm_std_ms": float(np.std(admm_arr, ddof=1)) if runs > 1 else 0.0,
        "admm_min_ms": float(np.min(admm_arr)),
        "admm_iters": admm_info["iters"],
        "admm_residual": admm_info["residual"],
        "admm_uncertain_bits": unc_admm,
        "speedup_mean": float(np.mean(admm_arr) / np.mean(trsd_arr)),
        "bit_diff_trsd_admm": bit_diff,
        "kdr_trsd_vs_kgroup": kdr_trsd,
        "kdr_admm_vs_kgroup": kdr_admm,
    }

    print(f"  TRSD : {result['trsd_mean_ms']:8.3f} ± {result['trsd_std_ms']:6.3f} ms  "
          f"(iters={result['trsd_iters']}, res={result['trsd_residual']:.4f}, "
          f"unc={result['trsd_uncertain_bits']})")
    print(f"  ADMM : {result['admm_mean_ms']:8.3f} ± {result['admm_std_ms']:6.3f} ms  "
          f"(iters={result['admm_iters']}, res={result['admm_residual']:.4f}, "
          f"unc={result['admm_uncertain_bits']})")
    print(f"  Speedup (ADMM/TRSD wall-time): {result['speedup_mean']:.1f}x")
    print(f"  Bit-diff TRSD-vs-ADMM (codeword): {bit_diff}/{2*n_main}")
    print(f"  KDR vs K_group: TRSD={kdr_trsd}/256, ADMM={kdr_admm}/256")

    # Persist
    rep_path = OUTPUT_DIR / f"16_benchmark_{exp}.txt"
    with rep_path.open("w") as fh:
        for k, v in result.items():
            fh.write(f"{k}={v}\n")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["exp1", "exp2", "exp3"], default=None)
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    targets = [args.exp] if args.exp else ["exp1", "exp2", "exp3"]
    rows = []
    for e in targets:
        rows.append(benchmark_exp(e, args.runs))

    print("\n=== Summary ===")
    print(f"  {'Exp':<6} {'N':>5} {'TRSD (ms)':>14} {'ADMM (ms)':>16} {'Speedup':>10} {'ADMM iters':>11}")
    print(f"  {'-'*6} {'-'*5} {'-'*14} {'-'*16} {'-'*10} {'-'*11}")
    for r in rows:
        if "error" in r:
            print(f"  {r['exp']:<6} ERROR: {r['error']}")
            continue
        print(f"  {r['exp']:<6} {r['N']:>5} "
              f"{r['trsd_mean_ms']:>6.3f} ± {r['trsd_std_ms']:>4.3f} "
              f"{r['admm_mean_ms']:>7.2f} ± {r['admm_std_ms']:>5.2f} "
              f"{r['speedup_mean']:>9.1f}x {r['admm_iters']:>11}")

    summary_path = OUTPUT_DIR / "16_benchmark_summary.csv"
    pd.DataFrame([r for r in rows if "error" not in r]).to_csv(summary_path, index=False)
    print(f"\n  Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
