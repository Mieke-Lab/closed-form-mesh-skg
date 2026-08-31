"""
09_alice_orchestrator.py
------------------------
End-to-end driver for Alice across the KeyCome Group-Key SKG pipeline.

Pipeline:
    01 Distribution Adjustment
    02 Channel Obfuscation
    03 Key Encoding & Encapsulation         (b_AB -> Bob, b_AC -> Charlie)
    .. wait for Bob / Charlie ADMM stage to publish their error indices ..
    06 Information Reconciliation           (parallel SCP push/pull)
    08 Privacy Amplification

Coordination after Stage 03 polls each peer's remote output directory until the
expected error-index files are present, with a configurable timeout.
"""
from __future__ import annotations
import os

import argparse
import importlib.util
import subprocess
import sys
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

STAGE_FILES = {
    "00": BASE_DIR / "00_session_seed.py",
    "01": BASE_DIR / "01_alice_distribution_adjustment.py",
    "02": BASE_DIR / "02_alice_channel_obfuscation.py",
    "03": BASE_DIR / "03_alice_key_encoding.py",
    "06": BASE_DIR / "06_alice_info_reconciliation.py",
    "08": BASE_DIR / "08_alice_privacy_amplification.py",
    "11": BASE_DIR / "11_eve_security_analysis.py",
}

PEERS = [
    {
        "name": "Bob",
        "host": os.environ.get("BOB_HOST", "bob@10.0.0.2"),
        "password": os.environ.get("BOB_PASS", ""),
        "remote_dir": "/home/bob/secret-key-generation-collection/magisterSeries/2026/IES_basedPTM_No3/channelProb/output",
        "remote_channelprob": "/home/bob/secret-key-generation-collection/magisterSeries/2026/IES_basedPTM_No3/channelProb",
        "orchestrator": "09_bob_orchestrator.py",
        "indicator": "05_bob_error_indices_{exp}.json",
    },
    {
        "name": "Charlie",
        "host": os.environ.get("CHARLIE_HOST", "charlie@10.0.0.3"),
        "password": os.environ.get("CHARLIE_PASS", ""),
        "remote_dir": "/home/charlie/secret-key-generation-collection/magisterSeries/2026/IES_basedPTM_No3/channelProb/output",
        "remote_channelprob": "/home/charlie/secret-key-generation-collection/magisterSeries/2026/IES_basedPTM_No3/channelProb",
        "orchestrator": "09_charlie_orchestrator.py",
        "indicator": "05_charlie_error_indices_{exp}.json",
    },
]


def _load(stage_key: str):
    """Dynamically import a stage module despite leading-digit filenames."""
    path = STAGE_FILES[stage_key]
    spec = importlib.util.spec_from_file_location(f"alice_stage_{stage_key}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ssh_cmd(host: str, password: str, command: str, timeout: int = 20) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["sshpass", "-p", password, "ssh", "-o", "StrictHostKeyChecking=no", host, command],
        capture_output=True, timeout=timeout,
    )


def remote_has_file(host: str, password: str, remote_path: str) -> bool:
    try:
        result = _ssh_cmd(host, password, f"test -f {remote_path}")
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def trigger_peers(exp: str) -> None:
    """SSH into each peer and launch its orchestrator in the background."""
    import os as _os
    env_prefix = ""
    if _os.environ.get("DISABLE_BKRR") == "1":
        env_prefix += "DISABLE_BKRR=1 "
    if _os.environ.get("DISABLE_CARR") == "1":
        env_prefix += "DISABLE_CARR=1 "
    nohup_env = f"env {env_prefix}".strip() if env_prefix else ""
    for peer in PEERS:
        cmd = (
            f"cd {peer['remote_channelprob']} && "
            f"nohup {nohup_env} python3 {peer['orchestrator']} --exp {exp} "
            f"> /tmp/{peer['name'].lower()}_skg_{exp}.log 2>&1 &"
        )
        try:
            _ssh_cmd(peer["host"], peer["password"], cmd, timeout=15)
            print(f"    [TRIGGER] {peer['name']} orchestrator started (log: /tmp/{peer['name'].lower()}_skg_{exp}.log)")
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            print(f"    [WARN] Could not trigger {peer['name']}: {exc}")


def wait_for_peers(exp: str, timeout_s: int, poll_s: int) -> bool:
    """Poll each peer until its error-index file appears or timeout elapses."""
    deadline = time.time() + timeout_s
    pending = {p["name"]: p for p in PEERS}
    print(f"    [WAIT] polling peers for error-index files (timeout={timeout_s}s) ...")
    while pending and time.time() < deadline:
        ready = []
        for name, peer in pending.items():
            remote_path = f"{peer['remote_dir']}/{peer['indicator'].format(exp=exp)}"
            if remote_has_file(peer["host"], peer["password"], remote_path):
                print(f"    [WAIT] {name} ready")
                ready.append(name)
        for name in ready:
            pending.pop(name)
        if pending:
            names = ", ".join(pending)
            elapsed = timeout_s - int(deadline - time.time())
            print(f"    [WAIT] {elapsed}s elapsed — still waiting: {names}")
            time.sleep(poll_s)
    if pending:
        print(f"    [WAIT] timed out waiting for: {', '.join(pending)}")
        return False
    print("    [WAIT] all peers ready")
    return True


def run(exp: str, wait_timeout: int, wait_poll: int, skip_wait: bool) -> None:
    t0 = time.time()
    print(f"=== Alice orchestrator start ({exp}) ===\n")

    _load("00").run(exp)
    _load("01").run(exp)
    _load("02").run(exp)
    _load("03").run(exp)

    if not skip_wait:
        trigger_peers(exp)
        if not wait_for_peers(exp, wait_timeout, wait_poll):
            print("[ERROR] Aborting before Stage 06 because peers are not ready.")
            return

    _load("06").run(exp)
    _load("08").run(exp)

    # Stage 11: Eve security analysis (runs on Alice without peer interaction)
    try:
        _load("11").run(exp)
    except Exception as exc:
        print(f"    [WARN] Stage 11 Eve analysis failed: {exc}")

    elapsed = time.time() - t0
    time_path = BASE_DIR / "output" / f"09_total_time_{exp}.txt"
    time_path.write_text(f"total_wall_seconds={elapsed:.3f}\n")

    print(f"=== Alice orchestrator done ({exp}) in {elapsed:.2f}s ===")
    final_key_path = BASE_DIR / "output" / f"08_alice_final_key_{exp}.txt"
    if final_key_path.is_file():
        print(f"    K_final stored at: {final_key_path}")
    print(f"    [TIME] {time_path.name} written (total={elapsed:.2f}s)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Alice end-to-end orchestrator")
    parser.add_argument("--exp", required=True,
                        choices=["exp1", "exp2", "exp3",
                                 "exp1_eve", "exp2_eve", "exp3_eve"])
    parser.add_argument("--wait-timeout", type=int, default=300,
                        help="Seconds to wait for peer error-index files (default 300)")
    parser.add_argument("--wait-poll", type=int, default=5,
                        help="Polling interval in seconds (default 5)")
    parser.add_argument("--skip-wait", action="store_true",
                        help="Skip Bob/Charlie polling (useful for offline tests)")
    args = parser.parse_args(argv)
    try:
        if "_eve" in args.exp:
            # Eve-only mode: just run Stage 11 security analysis
            print(f"=== Eve security analysis mode ({args.exp}) ===\n")
            _load("11").run(args.exp)
        else:
            run(args.exp, args.wait_timeout, args.wait_poll, args.skip_wait)
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
