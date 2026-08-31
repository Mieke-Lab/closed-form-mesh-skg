"""
15_verify_table3.py
-------------------
Independent verifier for paper Table III (NIST Randomness Test Results).

Reads `output/10_nist_results_exp{1,2,3}.csv` produced by Alice's Stage 10
(`10_nist_validation.py`) and cross-checks every published $p$-value cell
against the paper, plus the global PASS-rate claim (24/24 with
$p \ge 0.01$).

Paper Table III claims (rounded to 3 decimals):
    Test                          | S1    | S2    | S3
    Frequency (Monobit)           | 0.860 | 0.596 | 0.289
    Block Frequency (M=10)        | 0.327 | 0.084 | 0.084
    Runs                          | 0.077 | 0.879 | 0.648
    Longest Run of Ones           | 0.550 | 0.734 | 0.177
    Approx. Entropy (m=3)         | 0.288 | 0.443 | 0.738
    Cumulative Sums (Fwd)         | 0.655 | 0.431 | 0.500
    Cumulative Sums (Rev)         | 0.819 | 0.315 | 0.315
    Serial (m=3)                  | 0.386 | 0.437 | 0.732

Usage:
    python3 15_verify_table3.py            # all 3 exps
    python3 15_verify_table3.py --rerun    # re-run Stage 10 first then verify
    python3 15_verify_table3.py --exp exp2 # one scenario only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

# Paper Table III claimed p-values (S1=exp1, S2=exp2, S3=exp3).
# Key = canonical test name; columns = paper rounded p-values.
PAPER_TABLE_III = {
    "Frequency (Monobit)":         {"exp1": 0.860, "exp2": 0.596, "exp3": 0.289},
    "Block Frequency (M=10)":      {"exp1": 0.327, "exp2": 0.084, "exp3": 0.084},
    "Runs":                         {"exp1": 0.077, "exp2": 0.879, "exp3": 0.648},
    "Longest Run of Ones":          {"exp1": 0.550, "exp2": 0.734, "exp3": 0.177},
    "Approximate Entropy (m=3)":    {"exp1": 0.288, "exp2": 0.443, "exp3": 0.738},
    "Cumulative Sums (Forward)":    {"exp1": 0.655, "exp2": 0.431, "exp3": 0.500},
    "Cumulative Sums (Reverse)":    {"exp1": 0.819, "exp2": 0.315, "exp3": 0.315},
    "Serial (m=3)":                 {"exp1": 0.386, "exp2": 0.437, "exp3": 0.732},
}

P_THRESHOLD = 0.01
ROUND_TOL = 0.001   # paper rounds to 3 decimals -> allow 5e-4 rounding error


def rerun_stage10(exp: str) -> None:
    print(f"  [RERUN] python3 10_nist_validation.py --exp {exp}")
    proc = subprocess.run(
        ["python3", "10_nist_validation.py", "--exp", exp],
        cwd=str(BASE_DIR), capture_output=True, text=True
    )
    if proc.returncode != 0:
        print(f"    [WARN] non-zero exit: {proc.stderr[-200:]}")


def load_csv(exp: str) -> pd.DataFrame | None:
    path = OUTPUT_DIR / f"10_nist_results_{exp}.csv"
    if not path.is_file():
        print(f"  [WARN] missing {path}")
        return None
    return pd.read_csv(path)


def verify_exp(exp: str) -> dict:
    print(f"\n=== Verifying {exp} ===")
    df = load_csv(exp)
    if df is None:
        return {"exp": exp, "results": [], "fully_passed": False}

    # Build a lookup from test_name to (p_value, result)
    df_map = {}
    for _, row in df.iterrows():
        name = str(row["test_name"]).strip()
        df_map[name] = {"p": float(row["p_value"]),
                        "result": str(row["result"]).strip()}

    results = []
    for canonical, paper_row in PAPER_TABLE_III.items():
        claimed_p = paper_row[exp]
        # match by name (CSV uses same names)
        actual = df_map.get(canonical)
        if actual is None:
            # try suffix-stripped match
            for k, v in df_map.items():
                if k.split("(")[0].strip() == canonical.split("(")[0].strip():
                    actual = v
                    break
        if actual is None:
            results.append({"test": canonical, "claimed": claimed_p,
                            "measured": None, "pass_threshold": None,
                            "match_paper": False, "passed": False})
            continue
        measured_p = actual["p"]
        pass_threshold = measured_p >= P_THRESHOLD
        match_paper = abs(round(measured_p, 3) - claimed_p) <= ROUND_TOL
        results.append({
            "test": canonical, "claimed": claimed_p, "measured": measured_p,
            "pass_threshold": pass_threshold, "match_paper": match_paper,
            "passed": pass_threshold and match_paper,
        })

    # Print per-exp table
    print(f"  {'Test':<32} {'Measured':>10} {'Paper':>10} {'Match':>7} {'>=0.01':>8}")
    print(f"  {'-'*32} {'-'*10} {'-'*10} {'-'*7} {'-'*8}")
    n_total = n_match = n_pass = 0
    for r in results:
        n_total += 1
        m = "n/a" if r["measured"] is None else f"{r['measured']:.4f}"
        c = f"{r['claimed']:.3f}"
        ok_paper = "✓" if r["match_paper"] else "✗"
        ok_thresh = "✓" if r["pass_threshold"] else "✗"
        if r["match_paper"]: n_match += 1
        if r["pass_threshold"]: n_pass += 1
        print(f"  {r['test']:<32} {m:>10} {c:>10} {ok_paper:>7} {ok_thresh:>8}")
    print(f"  {'-'*32} {'-'*10} {'-'*10} {'-'*7} {'-'*8}")
    print(f"  {n_match}/{n_total} match paper rounding | {n_pass}/{n_total} pass p>=0.01")
    return {"exp": exp, "results": results,
            "fully_passed": n_pass == n_total and n_match == n_total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["exp1", "exp2", "exp3"],
                    help="single experiment (default all)")
    ap.add_argument("--rerun", action="store_true",
                    help="re-run Stage 10 (NIST validation) before verifying")
    args = ap.parse_args()

    targets = [args.exp] if args.exp else ["exp1", "exp2", "exp3"]
    if args.rerun:
        for e in targets:
            rerun_stage10(e)

    summary = []
    for e in targets:
        summary.append(verify_exp(e))

    # Overall
    total_cells = 0
    pass_threshold_cells = 0
    match_paper_cells = 0
    for s in summary:
        for r in s["results"]:
            total_cells += 1
            if r["pass_threshold"]:
                pass_threshold_cells += 1
            if r["match_paper"]:
                match_paper_cells += 1

    print("\n=== Table III Overall Verdict ===")
    print(f"  Paper rounding match : {match_paper_cells}/{total_cells}")
    print(f"  All p >= {P_THRESHOLD}      : {pass_threshold_cells}/{total_cells}")
    paper_claim = f"24/24 PASS, p>=0.01"
    if pass_threshold_cells == total_cells == 24 and match_paper_cells == 24:
        print(f"  Paper claim ({paper_claim}): CONFIRMED ✓")
        return 0
    print(f"  Paper claim ({paper_claim}): PARTIAL — see per-cell verdict")
    return 1


if __name__ == "__main__":
    sys.exit(main())
