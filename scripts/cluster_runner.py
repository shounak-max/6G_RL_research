"""
scripts/cluster_runner.py
=========================
Automation script to sync code, manage dependencies, run simulations,
and retrieve results & figures from the remote GPU cluster (10.0.24.7:2222).
"""

import os
import sys
import time
from pathlib import Path
import paramiko

REMOTE_HOST = "10.0.24.7"
REMOTE_PORT = 2222
REMOTE_USER = "piyush"
REMOTE_PASS = "2£K40d#N"
REMOTE_DIR = "6G_RL_research"

LOCAL_ROOT = Path(__file__).resolve().parent.parent


def get_ssh_client():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        REMOTE_HOST,
        port=REMOTE_PORT,
        username=REMOTE_USER,
        password=REMOTE_PASS,
        timeout=15,
    )
    return ssh


def check_remote_env(ssh):
    print("[*] Checking remote Python environment & CUDA...")
    stdin, stdout, stderr = ssh.exec_command(
        "nvidia-smi && python3 -c \"import torch; print('PyTorch version:', torch.__version__, 'CUDA available:', torch.cuda.is_available())\" 2>&1"
    )
    out = stdout.read().decode()
    print(out)

    # Check dependencies and install if missing
    reqs = ["gymnasium", "stable-baselines3", "scipy", "scikit-learn", "pyyaml", "matplotlib"]
    print("[*] Ensuring required packages are installed...")
    install_cmd = f"pip install --quiet --upgrade {' '.join(reqs)} 2>&1"
    stdin, stdout, stderr = ssh.exec_command(install_cmd)
    res = stdout.read().decode()
    if res.strip():
        print(res)
    print("[+] Remote environment ready.")


def sync_to_remote(ssh):
    print(f"[*] Syncing local repository to remote directory ~/{REMOTE_DIR}...")
    sftp = ssh.open_sftp()

    def remote_mkdir_p(remote_path):
        dirs = []
        head = remote_path
        while head and head != "/":
            dirs.append(head)
            head = os.path.dirname(head)
        for d in reversed(dirs):
            try:
                sftp.mkdir(d)
            except IOError:
                pass

    ignore_dirs = {".git", ".system_generated", "__pycache__", "venv", ".venv", ".idea", ".vscode"}
    ignore_extensions = {".pyc", ".log", ".tmp"}

    count = 0
    for root, dirs, files in os.walk(LOCAL_ROOT):
        # Filter ignore dirs
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        rel_dir = os.path.relpath(root, LOCAL_ROOT)
        if rel_dir == ".":
            remote_parent = REMOTE_DIR
        else:
            remote_parent = f"{REMOTE_DIR}/{rel_dir.replace(os.sep, '/')}"

        remote_mkdir_p(remote_parent)

        for f in files:
            ext = os.path.splitext(f)[1]
            if ext in ignore_extensions:
                continue
            local_file = os.path.join(root, f)
            remote_file = f"{remote_parent}/{f}"
            try:
                sftp.put(local_file, remote_file)
                count += 1
            except Exception as e:
                print(f"[!] Error uploading {f}: {e}")

    sftp.close()
    print(f"[+] Successfully synced {count} files to remote.")


def fetch_from_remote(ssh):
    print("[*] Fetching results and generated figures from remote cluster...")
    sftp = ssh.open_sftp()
    remote_dirs = [f"{REMOTE_DIR}/data", f"{REMOTE_DIR}/figures"]

    for r_dir in remote_dirs:
        try:
            files = sftp.listdir(r_dir)
            local_target_dir = LOCAL_ROOT / os.path.basename(r_dir)
            local_target_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                r_path = f"{r_dir}/{f}"
                l_path = local_target_dir / f
                try:
                    sftp.get(r_path, str(l_path))
                    print(f"    [+] Downloaded: {f} -> {l_path}")
                except Exception as e:
                    print(f"    [!] Failed to download {f}: {e}")
        except Exception as e:
            print(f"[!] Could not list remote directory {r_dir}: {e}")
    sftp.close()
    print("[+] Download complete.")


def run_remote_command(ssh, cmd, title):
    print("\n" + "=" * 75)
    print(f"[*] EXECUTING: {title}")
    print(f"    CMD: {cmd}")
    print("=" * 75)
    full_cmd = f"cd {REMOTE_DIR} && {cmd}"
    stdin, stdout, stderr = ssh.exec_command(full_cmd, get_pty=True)

    for line in iter(stdout.readline, ""):
        print(line, end="", flush=True)

    status = stdout.channel.recv_exit_status()
    print(f"\n[+] '{title}' finished with exit code {status}.")
    return status


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Remote GPU Cluster Runner")
    parser.add_argument("--sync-only", action="store_true", help="Sync files only")
    parser.add_argument("--fetch-only", action="store_true", help="Fetch results only")
    parser.add_argument("--run-all", action="store_true", help="Run full convergence, MARL stress, selector, and WUGNN")
    parser.add_argument("--timesteps", type=int, default=100000, help="Timesteps for convergence run")
    args = parser.parse_args()

    ssh = get_ssh_client()
    try:
        if args.fetch_only:
            fetch_from_remote(ssh)
            return

        check_remote_env(ssh)
        sync_to_remote(ssh)

        if args.sync_only:
            return

        # 1. Distributional Policy Selection Benchmark
        run_remote_command(
            ssh,
            "python3 pretraining/extra_trees_selector.py --benchmark-shift --use-distributional",
            "Experiment 3: Distributional Policy Selection Benchmark",
        )

        # 2. WUGNN Execution-Time Benchmark
        run_remote_command(
            ssh,
            "python3 scripts/wugnn_benchmark.py --n-list 10 20 50 100 --n-repeats 30",
            "Experiment 4: WUGNN vs Classical WMMSE Benchmark",
        )

        # 3. MARL Ablation Stress Test (Realistic Channel + 3 seeds)
        run_remote_command(
            ssh,
            "python3 agents/marl/marl_ablation_runner.py --steps 3000 --num-ues 8 --num-rbs 12 --num-episodes 3 --include-blockage --output-json data/marl_ablation_results.json",
            "Experiment 2: Realistic Channel MARL Stress-Testing",
        )

        # 4. High-Step Convergence Run (100k timesteps on GPU)
        run_remote_command(
            ssh,
            f"python3 scripts/convergence_run.py --sla-profile balanced --max-timesteps {args.timesteps} --num-ues 8 --num-rbs 12 --eval-episodes 5 --save-json data/convergence_results.json --figure-path figures/fig4_convergence_curve.png",
            f"Experiment 1: High-Step Convergence Run ({args.timesteps:,} steps)",
        )

        # Fetch all generated results and figures
        fetch_from_remote(ssh)

    finally:
        ssh.close()


if __name__ == "__main__":
    main()
