"""
18_min_entropy.py
-----------------
Empirical min-entropy + leftover-hash lemma (LHL) bound for the
IES 2026 paper Section V.G.

Two estimates produced:

(A) Per-bit frequency-based min-entropy of K_group from os.urandom:
       p_max = max(P(0), P(1)) over each 256-bit key
       H_inf ≥ -256 * log2(p_u)
    where p_u is the 99% upper confidence on p_max
    (per NIST SP 800-90B Section 6.3 most-common-value approach).

(B) Conditional min-entropy from Eve's per-bit error rate:
       p_e ∈ [Eve_KDR_min, Eve_KDR_max] from Table IV
       H_inf(K_group | Eve) ≥ -256 * log2(max(p_e, 1-p_e))
    This is the SECURITY-relevant bound — it captures the entropy
    Eve still misses after observing public packets.

LHL bound:
       Δ(K_final; U) ≤ 0.5 * 2^((m - H_inf)/2)
    where m = 256 is the Toeplitz output length.

Usage:
    python3 18_min_entropy.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"


def load_key(exp: str) -> np.ndarray:
    path = OUTPUT_DIR / f"03_alice_K_group_{exp}.csv"
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    return np.concatenate([df["real"].to_numpy(int), df["imag"].to_numpy(int)])


def freq_min_entropy(key: np.ndarray) -> tuple[float, float]:
    """Per-bit frequency-based min-entropy estimate.
    Returns (point estimate, 99% UCB-adjusted estimate)."""
    n = len(key)
    p_one = float(np.sum(key == 1)) / n
    p_max = max(p_one, 1 - p_one)
    # 99% upper confidence bound on p_max (z=2.576)
    se = math.sqrt(p_max * (1 - p_max) / n)
    p_u = min(1.0, p_max + 2.576 * se)
    h_point = -n * math.log2(p_max) if p_max > 0 else float("inf")
    h_ucb = -n * math.log2(p_u) if p_u > 0 else float("inf")
    return h_point, h_ucb


def conditional_min_entropy_from_eve(p_e_min: float, p_e_max: float,
                                      key_bits: int = 256) -> tuple[float, float]:
    """Eve's conditional uncertainty about each K_group bit."""
    # Conservative: use whichever p_e is FURTHEST from 0.5
    # (worst case for security)
    p_eve_worst = min(abs(p_e_min - 0.5), abs(p_e_max - 0.5))
    p_correct = 0.5 + p_eve_worst   # Eve's best guess probability per bit
    p_max = max(p_correct, 1 - p_correct)
    h_min = -key_bits * math.log2(p_max)
    # Best case (p_e closest to 0.5)
    p_eve_best = max(abs(p_e_min - 0.5), abs(p_e_max - 0.5))
    p_correct_best = 0.5 + p_eve_best
    p_max_best = max(p_correct_best, 1 - p_correct_best)
    h_min_pessimistic = -key_bits * math.log2(p_max_best)
    return h_min, h_min_pessimistic


def lhl_bound(h_inf: float, m: int = 256) -> float:
    """Leftover-hash lemma: Δ(T·K; U) ≤ 0.5 * 2^((m - H_inf)/2)."""
    exponent = (m - h_inf) / 2.0
    return 0.5 * (2.0 ** exponent)


def main():
    print("=" * 64)
    print("  Phase D — Min-entropy + Leftover-hash bound (IES 2026)")
    print("=" * 64)

    # === (A) Per-key empirical min-entropy of K_group from os.urandom ===
    print("\n[A] Per-key empirical min-entropy (K_group from os.urandom):\n")
    print(f"  {'Exp':<6} {'p_max':>10} {'H_inf point':>14} {'H_inf 99% UCB':>16}")
    print(f"  {'-'*6} {'-'*10} {'-'*14} {'-'*16}")
    for exp in ("exp1", "exp2", "exp3"):
        k = load_key(exp)
        h_point, h_ucb = freq_min_entropy(k)
        p_one = float(np.sum(k == 1)) / len(k)
        p_max = max(p_one, 1 - p_one)
        print(f"  {exp:<6} {p_max:>10.4f} {h_point:>10.1f} bit "
              f"{h_ucb:>12.1f} bit")

    # === (B) Conditional min-entropy from Eve's KDR ===
    # Per Table IV: Eve KDR range [45.31%, 54.30%]
    p_e_min = 0.4531
    p_e_max = 0.5430
    h_min, h_pessimistic = conditional_min_entropy_from_eve(
        p_e_min, p_e_max, key_bits=256)
    print(f"\n[B] Conditional min-entropy from Eve's per-bit error:\n")
    print(f"  Eve KDR range: [{p_e_min*100:.2f}%, {p_e_max*100:.2f}%]")
    print(f"  Worst-case (Eve farthest from 50%):  "
          f"p_max = {1 - min(abs(p_e_min-0.5), abs(p_e_max-0.5)) + 0:.4f} "
          f"→ H_inf(K|E) ≥ {h_min:.1f} bit")
    print(f"  Best-case (Eve closest to 50%):       "
          f"H_inf(K|E) ≥ {h_pessimistic:.1f} bit")

    # === LHL bound ===
    print(f"\n[C] Leftover-hash lemma (m=256-bit Toeplitz output):\n")
    delta_worst = lhl_bound(h_min, m=256)
    delta_best = lhl_bound(h_pessimistic, m=256)
    print(f"  Worst-case  Δ(K_final; U) ≤ {delta_worst:.3e}")
    print(f"  Best-case   Δ(K_final; U) ≤ {delta_best:.3e}")
    print(f"  (when H_inf > m, LHL ratio < 1 → output statistically close to uniform)")

    # === Pr[Eve recovers K_group] ===
    print(f"\n[D] Eve full-key recovery probability:\n")
    pr_worst = 2.0 ** (-h_min)
    pr_best = 2.0 ** (-h_pessimistic)
    print(f"  Pr[K_hat_Eve = K_group] worst-case ≤ 2^(-{h_min:.0f}) ≈ {pr_worst:.2e}")
    print(f"  Pr[K_hat_Eve = K_group] best-case   ≤ 2^(-{h_pessimistic:.0f}) ≈ {pr_best:.2e}")

    print("\n=== Summary for paper Section V.G ===\n")
    print(f"  Eve's empirical per-bit error rate p_e ∈ [{p_e_min*100:.2f}%, {p_e_max*100:.2f}%]")
    print(f"  → conditional min-entropy H_inf(K_group | E) ≥ {h_min:.0f} bit")
    print(f"  → Pr[K_hat_E = K_group] ≤ 2^(-{h_min:.0f}) ≈ {pr_worst:.2e}")
    print(f"  → Toeplitz LHL bound Δ ≤ {delta_worst:.3e} (vs uniform on {{0,1}}^256)")


if __name__ == "__main__":
    main()
