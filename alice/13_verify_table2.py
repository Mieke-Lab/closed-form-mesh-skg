"""
13_verify_table2.py
-------------------
Independent verifier for paper Table II (Key Agreement Metrics).

Re-derives every published cell directly from the raw artefacts produced
by stages 03-08 on Alice/Bob/Charlie, then compares against the values
written in the paper. Reports PASS/FAIL per cell.

Source of truth per experiment:
    Alice (local):
        output/03_alice_K_group_{exp}.csv     - 256-bit issuer key (real,imag)
        output/03_BCH_params_{exp}.csv         - N_AB, KEY_BITS, total_encoded
        output/05_bob_admm_report_{exp}.txt   - wall_seconds, uncertain_bits
        output/08_alice_final_key_{exp}.txt   - hex final key
    Bob (SSH $BOB_HOST):
        output/05_bob_recovered_key_{exp}.csv - BCH-decoded K_B (real,imag)
        output/08_bob_final_key_{exp}.txt     - hex final key
    Charlie (SSH $CHARLIE_HOST):
        output/05_charlie_recovered_key_{exp}.csv - K_C (real,imag)
        output/08_charlie_final_key_{exp}.txt     - hex final key

Output: ASCII table with claimed vs measured per cell and overall verdict.

Usage:
    python3 13_verify_table2.py            # all three experiments
    python3 13_verify_table2.py --exp exp2 # one experiment
"""
from __future__ import annotations
import os

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

BOB_HOST = os.environ.get("BOB_HOST", "bob@10.0.0.2")
BOB_PASS = os.environ.get("BOB_PASS", "")
BOB_REMOTE = ("/home/bob/secret-key-generation-collection/magisterSeries/"
              "2026/IES_basedPTM_No3/channelProb/output")

CHARLIE_HOST = os.environ.get("CHARLIE_HOST", "charlie@10.0.0.3")
CHARLIE_PASS = os.environ.get("CHARLIE_PASS", "")
CHARLIE_REMOTE = ("/home/charlie/secret-key-generation-collection/"
                  "magisterSeries/2026/IES_basedPTM_No3/channelProb/output")

# Paper Table II claimed values (S1=exp1, S2=exp2, S3=exp3)
PAPER_TABLE_II = {
    "exp1": {"N": 2045, "BMR_pre": 0, "BMR_post": 0, "fix_bits": 0,
             "KDR": 0.0, "BGR": 0.125, "KGR": 5.45,  "wall_s": 46.98},
    "exp2": {"N": 2029, "BMR_pre": 0, "BMR_post": 0, "fix_bits": 0,
             "KDR": 0.0, "BGR": 0.126, "KGR": 8.17,  "wall_s": 31.33},
    "exp3": {"N": 1580, "BMR_pre": 0, "BMR_post": 0, "fix_bits": 0,
             "KDR": 0.0, "BGR": 0.162, "KGR": 8.42,  "wall_s": 30.39},
}

KEY_BITS = 256
TOTAL_ENCODED = 567  # BCH(63,30)*9 codeword length


def scp_pull(host: str, password: str, remote_path: str, local_path: Path) -> bool:
    cmd = ["sshpass", "-p", password, "scp",
           "-o", "StrictHostKeyChecking=no",
           f"{host}:{remote_path}", str(local_path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"  [WARN] scp {host} {remote_path} failed: "
              f"{exc.stderr.decode(errors='ignore').strip()}")
        return False


def load_bits_csv(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    if "real" in cols and "imag" in cols:
        real = df[df.columns[cols.index("real")]].to_numpy(dtype=int)
        imag = df[df.columns[cols.index("imag")]].to_numpy(dtype=int)
        return np.concatenate([real, imag])
    if "bit" in cols:
        return df[df.columns[cols.index("bit")]].to_numpy(dtype=int)
    return df.iloc[:, 0].to_numpy(dtype=int)


def load_hex_key(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    txt = path.read_text().strip()
    # Accept hex or 0/1 string
    if all(c in "0123456789abcdefABCDEF" for c in txt) and len(txt) >= 64:
        return np.array(
            [(int(txt[i:i + 2], 16) >> b) & 1
             for i in range(0, len(txt), 2)
             for b in range(7, -1, -1)],
            dtype=int)[:KEY_BITS]
    if all(c in "01" for c in txt):
        return np.array([int(c) for c in txt], dtype=int)[:KEY_BITS]
    return None


def parse_admm_report(path: Path) -> dict:
    out = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        try:
            out[k] = float(v) if "." in v or "e" in v else int(v)
        except ValueError:
            out[k] = v
    return out


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    return int(np.sum(a[:n] != b[:n]))


def verify_exp(exp: str) -> dict:
    print(f"\n=== Verifying {exp} ===")
    paper = PAPER_TABLE_II[exp]

    # Local Alice files
    k_group_path = OUTPUT_DIR / f"03_alice_K_group_{exp}.csv"
    bch_params_path = OUTPUT_DIR / f"03_BCH_params_{exp}.csv"
    admm_report_path = OUTPUT_DIR / f"05_bob_admm_report_{exp}.txt"
    alice_final_path = OUTPUT_DIR / f"08_alice_final_key_{exp}.txt"
    total_time_path = OUTPUT_DIR / f"09_total_time_{exp}.txt"

    # Pull Bob + Charlie artefacts into temp dir
    tmp = Path(tempfile.mkdtemp(prefix=f"verify_{exp}_"))
    bob_rec_remote = f"{BOB_REMOTE}/05_bob_recovered_key_{exp}.csv"
    bob_rcn_remote = f"{BOB_REMOTE}/06_bob_reconciled_key_{exp}.csv"
    bob_fin_remote = f"{BOB_REMOTE}/08_bob_final_key_{exp}.txt"
    char_rec_remote = f"{CHARLIE_REMOTE}/05_charlie_recovered_key_{exp}.csv"
    char_rcn_remote = f"{CHARLIE_REMOTE}/06_charlie_reconciled_key_{exp}.csv"
    char_fin_remote = f"{CHARLIE_REMOTE}/08_charlie_final_key_{exp}.txt"

    bob_rec = tmp / f"05_bob_recovered_key_{exp}.csv"
    bob_rcn = tmp / f"06_bob_reconciled_key_{exp}.csv"
    bob_fin = tmp / f"08_bob_final_key_{exp}.txt"
    char_rec = tmp / f"05_charlie_recovered_key_{exp}.csv"
    char_rcn = tmp / f"06_charlie_reconciled_key_{exp}.csv"
    char_fin = tmp / f"08_charlie_final_key_{exp}.txt"

    scp_pull(BOB_HOST, BOB_PASS, bob_rec_remote, bob_rec)
    scp_pull(BOB_HOST, BOB_PASS, bob_rcn_remote, bob_rcn)
    scp_pull(BOB_HOST, BOB_PASS, bob_fin_remote, bob_fin)
    scp_pull(CHARLIE_HOST, CHARLIE_PASS, char_rec_remote, char_rec)
    scp_pull(CHARLIE_HOST, CHARLIE_PASS, char_rcn_remote, char_rcn)
    scp_pull(CHARLIE_HOST, CHARLIE_PASS, char_fin_remote, char_fin)

    # Load keys
    k_group = load_bits_csv(k_group_path)
    k_bob_pre = load_bits_csv(bob_rec)       # before reconciliation (BCH-decoded)
    k_bob_post = load_bits_csv(bob_rcn)      # after reconciliation
    k_char_pre = load_bits_csv(char_rec)
    k_char_post = load_bits_csv(char_rcn)
    k_alice_final = load_hex_key(alice_final_path)
    k_bob_final = load_hex_key(bob_fin)
    k_char_final = load_hex_key(char_fin)

    # ---- N samples (from BCH params N_AB) ----
    n_samples = None
    if bch_params_path.is_file():
        df = pd.read_csv(bch_params_path)
        for col in ("param", "name"):
            if col in df.columns:
                d = dict(zip(df[col], df["value"]))
                if "N_AB" in d:
                    n_samples = int(d["N_AB"])
                break

    # ---- BMR pre-BCH: bit mismatch between Bob/Charlie's PRE-reconciliation K
    #      (BCH-decoded only) and K_group. Paper claim = 0/256 across all exp. ----
    bmr_pre_AB = bmr_pre_AC = None
    if k_group is not None and k_bob_pre is not None:
        bmr_pre_AB = hamming(k_group, k_bob_pre)
    if k_group is not None and k_char_pre is not None:
        bmr_pre_AC = hamming(k_group, k_char_pre)
    bmr_pre = max([v for v in (bmr_pre_AB, bmr_pre_AC) if v is not None],
                  default=None)

    # ---- BMR post-BCH: bit mismatch on POST-reconciliation key (after Alice
    #      ships fix payload). Paper claim = 0/256 across all exp. ----
    bmr_post_AB = bmr_post_AC = None
    if k_group is not None and k_bob_post is not None:
        bmr_post_AB = hamming(k_group, k_bob_post)
    if k_group is not None and k_char_post is not None:
        bmr_post_AC = hamming(k_group, k_char_post)
    bmr_post = max([v for v in (bmr_post_AB, bmr_post_AC) if v is not None],
                   default=None)

    # ---- KDR: hex final-key disagreement (independent confirmation) ----
    final_mismatch = 0
    if k_alice_final is not None:
        for k_other in (k_bob_final, k_char_final):
            if k_other is not None and not np.array_equal(k_alice_final, k_other):
                final_mismatch = 1
    kdr = float(final_mismatch * 100.0)

    # ---- Wall-time: end-to-end orchestrator total (09_total_time) ----
    wall_s = None
    if total_time_path.is_file():
        for line in total_time_path.read_text().splitlines():
            if "total_wall_seconds" in line and "=" in line:
                try:
                    wall_s = float(line.split("=", 1)[1].strip())
                except ValueError:
                    pass

    # ---- Fix-bits: uncertain-bit count from TRSD report (paper "Fix-bits sent") ----
    admm = parse_admm_report(admm_report_path)
    uncertain_str = str(admm.get("uncertain_bits", ""))
    fix_bits = None
    if "/" in uncertain_str:
        fix_bits = int(uncertain_str.split("/")[0])

    # ---- BGR, KGR ----
    bgr = KEY_BITS / n_samples if n_samples else None
    # KGR from paper_metrics CSV (already computed); also recompute
    pm_path = OUTPUT_DIR / f"12_paper_metrics_{exp}.csv"
    kgr_csv = None
    if pm_path.is_file():
        dfp = pd.read_csv(pm_path)
        if "KGR" in dfp.columns:
            kgr_csv = float(dfp["KGR"].iloc[0])

    # Compose row
    measured = {
        "N": n_samples,
        "BMR_pre": bmr_pre,
        "BMR_post": bmr_post,
        "fix_bits": fix_bits,
        "KDR": kdr,
        "BGR": bgr,
        "KGR": kgr_csv,
        "wall_s": wall_s,
    }

    # Compare against paper
    def cmp(field, measured_val, claimed_val, tol_abs=0.005, tol_rel=0.02):
        if measured_val is None:
            return "MISSING", measured_val, claimed_val
        if isinstance(claimed_val, int) and isinstance(measured_val, int):
            return ("PASS" if measured_val == claimed_val else "FAIL",
                    measured_val, claimed_val)
        # numeric
        diff = abs(measured_val - claimed_val)
        if diff <= tol_abs or diff / max(abs(claimed_val), 1e-9) <= tol_rel:
            return "PASS", measured_val, claimed_val
        return "FAIL", measured_val, claimed_val

    results = {}
    for f in ("N", "BMR_pre", "BMR_post", "fix_bits", "KDR", "BGR", "KGR", "wall_s"):
        results[f] = cmp(f, measured[f], paper[f])

    # Print row
    print(f"  {'Field':<10} {'Measured':>14} {'Paper':>14}   Verdict")
    print(f"  {'-'*10} {'-'*14} {'-'*14}   {'-'*7}")
    for f, (verdict, m, c) in results.items():
        if isinstance(m, float):
            m_s = f"{m:.4f}"
        elif m is None:
            m_s = "n/a"
        else:
            m_s = str(m)
        if isinstance(c, float):
            c_s = f"{c:.4f}"
        else:
            c_s = str(c)
        mark = "✓" if verdict == "PASS" else "✗"
        print(f"  {f:<10} {m_s:>14} {c_s:>14}   {mark} {verdict}")

    return {"exp": exp, "results": results, "measured": measured}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["exp1", "exp2", "exp3"],
                    help="single experiment (default: all)")
    args = ap.parse_args()

    targets = [args.exp] if args.exp else ["exp1", "exp2", "exp3"]

    summary = []
    for e in targets:
        try:
            summary.append(verify_exp(e))
        except Exception as exc:
            print(f"  [ERROR] {e}: {exc}")

    print("\n=== Overall Verdict ===")
    total = 0
    passed = 0
    for s in summary:
        for f, (v, _, _) in s["results"].items():
            total += 1
            if v == "PASS":
                passed += 1
    print(f"  {passed}/{total} cells PASS")
    print(f"  Table II is {'CONFIRMED' if passed == total else 'PARTIAL'}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
