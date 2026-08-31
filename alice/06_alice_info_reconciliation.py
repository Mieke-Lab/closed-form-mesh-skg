"""
06_alice_info_reconciliation.py
-------------------------------
Stage 06 of the KeyCome Group-Key SKG pipeline (Alice).

Pulls the error-prone bit-index lists produced by Bob and Charlie via SSH,
reconstructs the relevant sub-matrices of the AB / AC circulant obfuscation
matrices, recomputes the partial encapsulations b_AB_fix and b_AC_fix, and
ships them back to the corresponding peers. Both peers are handled in parallel
using a thread pool so reconciliation latency is bounded by the slower side.

Inputs:
    channelProb/output/03_alice_K_group_{exp}.csv
    channelProb/output/02_alice_matrix_AB_{exp}.csv
    channelProb/output/02_alice_matrix_AC_{exp}.csv
    channelProb/output/03_BCH_params_{exp}.csv
    Remote: 06_bob_error_indices_{exp}.json    (pulled from Bob)
    Remote: 06_charlie_error_indices_{exp}.json (pulled from Charlie)

Outputs:
    channelProb/output/06_alice_done_{exp}.txt
    Remote: 06_alice_b_AB_fix_{exp}.csv + 06_alice_sub_indices_AB_{exp}.json
    Remote: 06_alice_b_AC_fix_{exp}.csv + 06_alice_sub_indices_AC_{exp}.json
"""
from __future__ import annotations
import os

import argparse
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import galois
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

PEERS = {
    "AB": {
        "name": "Bob",
        "host": os.environ.get("BOB_HOST", "bob@10.0.0.2"),
        "password": os.environ.get("BOB_PASS", ""),
        "remote_dir": "/home/bob/secret-key-generation-collection/magisterSeries/2026/IES_basedPTM_No3/channelProb/output",
        "fix_csv_tpl": "06_alice_b_AB_fix_{exp}.csv",
        "sub_idx_tpl": "06_alice_sub_indices_AB_{exp}.json",
        # Bob Stage 06 pushes the indices file to Alice; poll locally.
        "pull_remote_indices": False,
        "local_indices_tpl": "06_bob_error_indices_{exp}.json",
        # Bob Stage 06 expects (index, bit) direct corrections, not sub-ADMM encapsulation.
        "fix_format": "bits",
    },
    "AC": {
        "name": "Charlie",
        "host": os.environ.get("CHARLIE_HOST", "charlie@10.0.0.3"),
        "password": os.environ.get("CHARLIE_PASS", ""),
        "remote_dir": "/home/charlie/secret-key-generation-collection/magisterSeries/2026/IES_basedPTM_No3/channelProb/output",
        "fix_csv_tpl": "06_alice_charlie_sub_encapsulation_{exp}.csv",
        "sub_idx_tpl": "06_alice_charlie_sub_indices_{exp}.json",
        # Charlie has no sshpass; Alice pulls Stage 05 indices directly.
        "pull_remote_indices": True,
        "remote_indices_tpl": "05_charlie_error_indices_{exp}.json",
        # Send direct (index, bit) corrections — Charlie patches its interleaved estimate.
        "fix_format": "bits",
    },
}

LOCAL_PUSH_TIMEOUT_S = 90.0
LOCAL_PUSH_POLL_S = 2.0


def scp_pull(remote_host: str, password: str, remote_path: str, local_path: Path) -> bool:
    cmd = [
        "sshpass", "-p", password,
        "scp", "-o", "StrictHostKeyChecking=no",
        f"{remote_host}:{remote_path}", str(local_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="ignore") if exc.stderr else ""
        print(f"    [WARN] scp pull failed for {remote_path}: {stderr.strip()}")
        return False
    except FileNotFoundError:
        print("    [WARN] sshpass/scp not available locally")
        return False


def scp_push(local_files: list[Path], host: str, password: str, dest: str) -> bool:
    cmd = [
        "sshpass", "-p", password,
        "scp", "-o", "StrictHostKeyChecking=no",
        *[str(p) for p in local_files],
        f"{host}:{dest}/",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="ignore") if exc.stderr else ""
        print(f"    [WARN] scp push to {host} failed: {stderr.strip()}")
        return False
    except FileNotFoundError:
        print("    [WARN] sshpass/scp not available locally")
        return False


def encode_for_padding(k_group: np.ndarray, bch_n: int, bch_k: int) -> np.ndarray:
    """Reproduce the same BCH-encoded + interleaved bit stream from K_group."""
    n_blocks = math.ceil(k_group.shape[0] / bch_k)
    bch = galois.BCH(bch_n, bch_k)
    blocks = []
    for b in range(n_blocks):
        start = b * bch_k
        end = min(start + bch_k, k_group.shape[0])
        block = k_group[start:end]
        if len(block) < bch_k:
            block = np.concatenate([block, np.zeros(bch_k - len(block), dtype=int)])
        codeword = bch.encode(galois.GF2(block))
        blocks.append(np.array(codeword, dtype=int))
    all_encoded = np.concatenate(blocks)
    return all_encoded.reshape(n_blocks, bch_n).T.flatten()


def reconcile_peer(label: str, exp: str, k_group_bits: np.ndarray,
                   first_row: np.ndarray, total_encoded: int,
                   interleaved: np.ndarray) -> tuple[str, bool, int]:
    peer = PEERS[label]
    name = peer["name"]
    print(f"--- Reconciling with {name} ({label}) ---")

    # 1. Obtain error-index list from peer.
    if peer.get("pull_remote_indices"):
        # Alice pulls from peer's remote dir (peer has no sshpass to push).
        remote_file = peer["remote_indices_tpl"].format(exp=exp)
        remote_path = f"{peer['remote_dir']}/{remote_file}"
        local_idx = OUTPUT_DIR / f"06_{name.lower()}_error_indices_{exp}.json"
        if not scp_pull(peer["host"], peer["password"], remote_path, local_idx):
            print(f"    [{name}] failed to pull error indices from remote")
            return label, False, 0
        print(f"    [{name}] pulled {remote_file} from remote")
    else:
        # Peer pushes the file here; poll until it arrives.
        local_idx = OUTPUT_DIR / peer["local_indices_tpl"].format(exp=exp)
        deadline = time.time() + LOCAL_PUSH_TIMEOUT_S
        while not local_idx.is_file() and time.time() < deadline:
            elapsed = int(time.time() - (deadline - LOCAL_PUSH_TIMEOUT_S))
            print(f"    [{name}] waiting for error-index push ... {elapsed}s")
            time.sleep(LOCAL_PUSH_POLL_S)
        if not local_idx.is_file():
            print(f"    [{name}] timeout: {local_idx.name} never arrived")
            return label, False, 0

    try:
        with local_idx.open() as f:
            raw = json.load(f)
        # Support both flat list and dict-with-indices-key formats.
        error_indices = raw["indices"] if isinstance(raw, dict) else raw
    except Exception as exc:
        print(f"    [{name}] failed to parse {local_idx}: {exc}")
        return label, False, 0

    # Keep only indices that fall inside the BCH-encoded region.
    error_indices = sorted({int(i) for i in error_indices if 0 <= int(i) < total_encoded})
    print(f"    [{name}] {len(error_indices)} error-prone indices in info region")

    n = first_row.shape[0]

    fix_csv_name = peer["fix_csv_tpl"].format(exp=exp)
    sub_idx_name = peer["sub_idx_tpl"].format(exp=exp)

    if not error_indices:
        # Nothing to fix: send empty placeholders so peer knows reconciliation is a no-op.
        empty_csv = OUTPUT_DIR / fix_csv_name
        empty_idx = OUTPUT_DIR / sub_idx_name
        pd.DataFrame({"Real": [], "Imag": []}).to_csv(empty_csv, index=False)
        with empty_idx.open("w") as f:
            json.dump([], f)
        ok = scp_push([empty_csv, empty_idx], peer["host"], peer["password"], peer["remote_dir"])
        return label, ok, 0

    # 2. Reconstruct sub-matrix Pa' from circulant first row.
    matrix_indices = sorted({idx if idx < n else idx - n for idx in error_indices})
    idx_arr = np.array(matrix_indices, dtype=int)
    diff = (idx_arr[:, None] - idx_arr[None, :]) % n
    pa_sub = first_row[diff]

    # 3. Sub-vector of x_tilde (real + j*imag) at those indices.
    x_full = np.zeros(2 * n, dtype=int)
    x_full[:total_encoded] = interleaved
    real_part = x_full[:n][idx_arr].astype(float)
    imag_part = x_full[n:][idx_arr].astype(float)
    x_sub = real_part + 1j * imag_part

    fix_csv = OUTPUT_DIR / fix_csv_name
    sub_idx = OUTPUT_DIR / sub_idx_name

    if peer.get("fix_format") == "bits":
        # Send direct (index, bit) corrections — peer re-encodes and patches.
        bits_data = [{"index": int(idx), "bit": int(x_full[idx])} for idx in error_indices]
        pd.DataFrame(bits_data).to_csv(fix_csv, index=False)
        with sub_idx.open("w") as f:
            json.dump(error_indices, f)
    else:
        # 4. Re-encapsulate via direct mat-vec — peer runs sub-ADMM to recover bits.
        b_fix = pa_sub @ x_sub
        pd.DataFrame({"Real": np.real(b_fix), "Imag": np.imag(b_fix)}).to_csv(fix_csv, index=False)
        with sub_idx.open("w") as f:
            json.dump(matrix_indices, f)

    # 5. Push back to peer.
    ok = scp_push([fix_csv, sub_idx], peer["host"], peer["password"], peer["remote_dir"])
    print(f"    [{name}] fix sent ({len(matrix_indices)} indices) -> {'OK' if ok else 'FAIL'}")
    return label, ok, len(matrix_indices)


def run(exp: str) -> None:
    print(f"--- Stage 06: Alice Information Reconciliation ({exp}) ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load shared artefacts.
    k_group_path = OUTPUT_DIR / f"03_alice_K_group_{exp}.csv"
    bch_path = OUTPUT_DIR / f"03_BCH_params_{exp}.csv"
    if not k_group_path.is_file() or not bch_path.is_file():
        raise FileNotFoundError("K_group or BCH params missing -- run Stage 03 first.")

    df_k = pd.read_csv(k_group_path)
    k_group_bits = np.concatenate([df_k["real"].astype(int).to_numpy(),
                                   df_k["imag"].astype(int).to_numpy()])

    df_bch = pd.read_csv(bch_path)
    bch_dict = dict(zip(df_bch["param"], df_bch["value"].astype(int)))
    bch_n = bch_dict["BCH_N"]
    bch_k = bch_dict["BCH_K"]
    total_encoded = bch_dict["total_encoded_bits"]

    interleaved = encode_for_padding(k_group_bits, bch_n, bch_k)

    matrix_paths = {
        "AB": OUTPUT_DIR / f"02_alice_matrix_AB_{exp}.csv",
        "AC": OUTPUT_DIR / f"02_alice_matrix_AC_{exp}.csv",
    }
    first_rows: dict[str, np.ndarray] = {}
    for label, p in matrix_paths.items():
        if not p.is_file():
            raise FileNotFoundError(f"Stage 02 output missing: {p}")
        first_rows[label] = pd.read_csv(p)["first_row"].to_numpy(dtype=float)

    # Run AB and AC reconciliations in parallel threads.
    results: dict[str, tuple[bool, int]] = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(reconcile_peer, label, exp, k_group_bits,
                        first_rows[label], total_encoded, interleaved): label
            for label in PEERS
        }
        for fut in as_completed(futures):
            label, ok, count = fut.result()
            results[label] = (ok, count)

    done_path = OUTPUT_DIR / f"06_alice_done_{exp}.txt"
    with done_path.open("w") as f:
        for label, (ok, count) in results.items():
            f.write(f"{label}\t{'OK' if ok else 'FAIL'}\t{count}\n")
    print(f"    [DONE] summary -> {done_path}")
    print("Stage 06 complete.\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Alice Stage 06 - information reconciliation")
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
