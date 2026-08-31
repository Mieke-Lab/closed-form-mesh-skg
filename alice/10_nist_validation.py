"""
10_nist_validation.py
---------------------
NIST SP 800-22 randomness tests on the Alice final key (256-bit).

Implements 8 tests from NIST SP 800-22 Rev. 1a directly in Python so results
are reproducible and independent of the pre-compiled binary (which expects a
specific key length).  The 256-bit key is tested as two independent 128-bit
blocks; per-test p-value is the minimum of the two blocks (conservative).

Tests implemented (all valid for n=128 bits):
  1. Frequency (Monobit)
  2. Block Frequency (M=10)
  3. Runs
  4. Longest Run of Ones
  5. Approximate Entropy (m=3)
  6. Cumulative Sums (Forward)
  7. Cumulative Sums (Reverse)
  8. Serial (m=3)

PASS threshold: p >= 0.01  (NIST SP 800-22, Section 4.2.1)

Output:
    channelProb/output/10_nist_results_{exp}.csv  (test_name, p_value, result)
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

PASS_THRESHOLD = 0.01
BLOCK_M = 10   # Block Frequency block size
APEN_M  = 3    # Approximate Entropy pattern length
SERIAL_M = 3   # Serial test pattern length


# ── helpers ─────────────────────────────────────────────────────────────────

def igamc(a: float, x: float) -> float:
    """Upper incomplete gamma: Q(a, x) = igamc(a, x) via scipy."""
    return special.gammaincc(a, x)


def erfc(x: float) -> float:
    return math.erfc(x)


# ── individual tests ─────────────────────────────────────────────────────────

def test_frequency(bits: np.ndarray) -> float:
    """NIST 2.1 Frequency (Monobit) Test."""
    n = len(bits)
    s = np.sum(2 * bits.astype(int) - 1)
    s_obs = abs(s) / math.sqrt(n)
    return erfc(s_obs / math.sqrt(2))


def test_block_frequency(bits: np.ndarray, M: int = BLOCK_M) -> float:
    """NIST 2.2 Block Frequency Test."""
    n = len(bits)
    N = n // M
    if N < 1:
        return float("nan")
    chi2 = 0.0
    for i in range(N):
        block = bits[i * M:(i + 1) * M]
        pi_i = np.sum(block) / M
        chi2 += (pi_i - 0.5) ** 2
    chi2 *= 4 * M
    return igamc(N / 2.0, chi2 / 2.0)


def test_runs(bits: np.ndarray) -> float:
    """NIST 2.3 Runs Test."""
    n = len(bits)
    pi = np.sum(bits) / n
    if abs(pi - 0.5) >= 2 / math.sqrt(n):
        return 0.0  # pre-test fails → p=0
    v = 1 + np.sum(bits[:-1] != bits[1:])  # number of runs
    num = abs(v - 2 * n * pi * (1 - pi))
    den = 2 * math.sqrt(2 * n) * pi * (1 - pi)
    return erfc(num / den)


def test_longest_run(bits: np.ndarray) -> float:
    """NIST 2.4 Longest Run of Ones in a Block (n=128 table)."""
    n = len(bits)
    # For n=128: M=8, K=3, N=16, boundaries [1,2,3,4]
    M, N, K = 8, 16, 3
    bounds = [1, 2, 3, 4]
    pi = [0.2148, 0.3672, 0.2305, 0.1875]  # NIST table for n=128, M=8

    def max_run(blk):
        mx = cur = 0
        for b in blk:
            cur = cur + 1 if b == 1 else 0
            mx = max(mx, cur)
        return mx

    v = [0] * (K + 1)
    for i in range(N):
        block = bits[i * M:(i + 1) * M]
        r = max_run(block)
        if r <= bounds[0]:
            v[0] += 1
        elif r <= bounds[1]:
            v[1] += 1
        elif r <= bounds[2]:
            v[2] += 1
        else:
            v[3] += 1

    chi2 = sum((v[j] - N * pi[j]) ** 2 / (N * pi[j]) for j in range(K + 1))
    return igamc(K / 2.0, chi2 / 2.0)


def _psi_sq(bits: np.ndarray, m: int) -> float:
    """Helper for Approximate Entropy and Serial: compute psi_m^2."""
    n = len(bits)
    extended = np.concatenate([bits, bits[:m - 1]])  # wrap-around
    counts: dict[tuple, int] = {}
    for i in range(n):
        pattern = tuple(extended[i:i + m])
        counts[pattern] = counts.get(pattern, 0) + 1
    return sum(c ** 2 for c in counts.values()) * (2 ** m / n) - n


def test_approx_entropy(bits: np.ndarray, m: int = APEN_M) -> float:
    """NIST 2.12 Approximate Entropy Test."""
    n = len(bits)
    phi = []
    for mm in (m, m + 1):
        extended = np.concatenate([bits, bits[:mm - 1]])
        counts: dict[tuple, int] = {}
        for i in range(n):
            pat = tuple(int(x) for x in extended[i:i + mm])
            counts[pat] = counts.get(pat, 0) + 1
        phi.append(sum(c / n * math.log(c / n) for c in counts.values()))
    apen = phi[0] - phi[1]
    chi2 = 2 * n * (math.log(2) - apen)
    return igamc(2 ** (m - 1), chi2 / 2.0)


def test_cusum_forward(bits: np.ndarray) -> float:
    """NIST 2.13 Cumulative Sums Test (Forward)."""
    return _cusum(bits, forward=True)


def test_cusum_reverse(bits: np.ndarray) -> float:
    """NIST 2.13 Cumulative Sums Test (Reverse)."""
    return _cusum(bits, forward=False)


def _cusum(bits: np.ndarray, forward: bool) -> float:
    n = len(bits)
    x = 2 * bits.astype(int) - 1
    if not forward:
        x = x[::-1]
    s = np.cumsum(x)
    z = int(np.max(np.abs(s)))
    if z == 0:
        return 1.0
    # Sum series (NIST SP 800-22 eq 2.13.3)
    k_lo = int((-n / z + 1) / 4)
    k_hi = int((n / z - 1) / 4)
    sum1 = sum(
        special.ndtr((4 * k + 1) * z / math.sqrt(n)) -
        special.ndtr((4 * k - 1) * z / math.sqrt(n))
        for k in range(k_lo, k_hi + 1)
    )
    k_lo2 = int((-n / z - 3) / 4)
    k_hi2 = int((n / z - 1) / 4)
    sum2 = sum(
        special.ndtr((4 * k + 3) * z / math.sqrt(n)) -
        special.ndtr((4 * k + 1) * z / math.sqrt(n))
        for k in range(k_lo2, k_hi2 + 1)
    )
    return 1.0 - sum1 + sum2


def test_serial(bits: np.ndarray, m: int = SERIAL_M) -> float:
    """NIST 2.11 Serial Test (returns min p of two statistics)."""
    n = len(bits)
    psi_m   = _psi_sq(bits, m)
    psi_m1  = _psi_sq(bits, m - 1) if m >= 2 else 0.0
    psi_m2  = _psi_sq(bits, m - 2) if m >= 3 else 0.0
    d1 = psi_m - psi_m1
    d2 = psi_m - 2 * psi_m1 + psi_m2
    p1 = igamc(2 ** (m - 2), d1 / 2.0)
    p2 = igamc(2 ** (m - 3), d2 / 2.0) if m >= 3 else 1.0
    return min(p1, p2)


# ── test registry ────────────────────────────────────────────────────────────

TESTS = [
    ("Frequency (Monobit)",        test_frequency),
    ("Block Frequency (M=10)",     test_block_frequency),
    ("Runs",                       test_runs),
    ("Longest Run of Ones",        test_longest_run),
    ("Approximate Entropy (m=3)",  test_approx_entropy),
    ("Cumulative Sums (Forward)",  test_cusum_forward),
    ("Cumulative Sums (Reverse)",  test_cusum_reverse),
    ("Serial (m=3)",               test_serial),
]


# ── main ─────────────────────────────────────────────────────────────────────

def hex_to_bits(hex_str: str) -> np.ndarray:
    n = len(hex_str) * 4
    val = int(hex_str, 16)
    return np.array([(val >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.uint8)


def run_tests_on_block(bits: np.ndarray) -> dict[str, float]:
    results = {}
    for name, fn in TESTS:
        try:
            p = float(fn(bits))
        except Exception as e:
            p = float("nan")
        results[name] = p
    return results


def print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 68)
    print("  NIST SP 800-22 Results (Python implementation, n=128 per block)")
    print("=" * 68)
    print(f"  {'Test':<38} {'p-value':>9}  Result")
    print("  " + "-" * 65)
    for r in rows:
        status = r["result"]
        p_str = f"{r['p_value']:.6f}" if not math.isnan(r["p_value"]) else "   N/A  "
        print(f"  {r['test_name']:<38} {p_str:>9}  {status}")
    print("=" * 68)
    n_pass = sum(1 for r in rows if r["result"] == "PASS")
    print(f"  {n_pass}/{len(rows)} tests PASSED (threshold p >= {PASS_THRESHOLD})\n")


def run(exp: str) -> None:
    print(f"--- Stage 10: NIST Validation ({exp}) ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    key_path = OUTPUT_DIR / f"08_alice_final_key_{exp}.txt"
    if not key_path.is_file():
        raise FileNotFoundError(f"Final key missing: {key_path} — run Stage 08 first.")

    hex_key = key_path.read_text().strip()
    bits = hex_to_bits(hex_key)
    n_total = len(bits)
    print(f"    [KEY] {n_total}-bit key loaded ({len(hex_key)} hex chars)")

    if n_total < 128:
        raise ValueError(f"Key too short ({n_total} bits); need >= 128.")

    # Run tests on each 128-bit block, report min p-value (conservative)
    block1 = bits[:128]
    block2 = bits[128:256] if n_total >= 256 else block1

    r1 = run_tests_on_block(block1)
    r2 = run_tests_on_block(block2)

    rows = []
    for name, _ in TESTS:
        p1, p2 = r1[name], r2[name]
        p = min(p for p in (p1, p2) if not math.isnan(p)) if not (math.isnan(p1) and math.isnan(p2)) else float("nan")
        rows.append({
            "test_name": name,
            "p_value": round(p, 6) if not math.isnan(p) else float("nan"),
            "p_block1": round(p1, 6),
            "p_block2": round(p2, 6),
            "result": "PASS" if (not math.isnan(p) and p >= PASS_THRESHOLD) else "FAIL",
        })

    print_table(rows)

    out_path = OUTPUT_DIR / f"10_nist_results_{exp}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"    [OK] {out_path.name} written ({len(rows)} tests)")
    print("Stage 10 complete.\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NIST SP 800-22 validation on Alice final key")
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
