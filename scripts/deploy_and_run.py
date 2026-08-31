import os
import sys
import time
from pathlib import Path
import paramiko

REMOTE_HOST = "10.0.24.7"
REMOTE_PORT = 2222
REMOTE_USER = "piyush"
REMOTE_PASS = "2£K40d#N"
REMOTE_DIR = "/home/piyush/6G_RL_research"
VENV_DIR = "/home/piyush/env_6g"
LOCAL_ROOT = Path(__file__).resolve().parent.parent


def get_client():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        REMOTE_HOST,
        port=REMOTE_PORT,
        username=REMOTE_USER,
        password=REMOTE_PASS,
        timeout=15,
    )
    ssh.get_transport().set_keepalive(15)
    return ssh


def run_cmd(ssh, cmd, title=""):
    if title:
        print("\n" + "=" * 70)
        print(f"[*] {title}")
        print(f"    CMD: {cmd}")
        print("=" * 70)
    
    transport = ssh.get_transport()
    channel = transport.open_session()
    channel.get_pty()
    channel.exec_command(cmd)
    
    output = []
    while True:
        if channel.recv_ready():
            data = channel.recv(4096).decode('utf-8', errors='replace')
            sys.stdout.write(data)
            sys.stdout.flush()
            output.append(data)
        if channel.exit_status_ready():
            # Flush remaining
            while channel.recv_ready():
                data = channel.recv(4096).decode('utf-8', errors='replace')
                sys.stdout.write(data)
                sys.stdout.flush()
                output.append(data)
            break
        time.sleep(0.1)
        
    status = channel.recv_exit_status()
    channel.close()
    return status, "".join(output)


def setup_remote_env(ssh):
    print("[*] Setting up Python virtual environment on remote cluster...")
    # Create venv if not exists
    cmd = f"""
    if [ ! -d "{VENV_DIR}" ]; then
        python3 -m venv {VENV_DIR}
    fi
    source {VENV_DIR}/bin/activate
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu118 || pip install torch
    pip install gymnasium stable-baselines3 scipy scikit-learn pyyaml matplotlib
    """
    status, _ = run_cmd(ssh, cmd, "Setting up Remote Virtual Environment")
    if status != 0:
        print("[!] Virtual environment setup returned non-zero status.")
    else:
        print("[+] Remote virtual environment ready.")


def sync_code(ssh):
    print(f"[*] Uploading code to {REMOTE_DIR}...")
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

    ignore_dirs = {".git", ".system_generated", "__pycache__", "venv", ".venv", ".idea", ".vscode", "runs"}
    ignore_extensions = {".pyc", ".log", ".tmp"}
    
    count = 0
    for root, dirs, files in os.walk(LOCAL_ROOT):
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
    print(f"[+] Synced {count} files to cluster.")


def download_results(ssh):
    print("[*] Downloading results & generated figures from cluster...")
    sftp = ssh.open_sftp()
    for subdir in ["data", "figures"]:
        r_dir = f"{REMOTE_DIR}/{subdir}"
        l_dir = LOCAL_ROOT / subdir
        l_dir.mkdir(parents=True, exist_ok=True)
        try:
            for f in sftp.listdir(r_dir):
                r_path = f"{r_dir}/{f}"
                l_path = l_dir / f
                try:
                    sftp.get(r_path, str(l_path))
                    print(f"    [+] Retrieved: {subdir}/{f}")
                except Exception as e:
                    print(f"    [!] Error getting {f}: {e}")
        except Exception as e:
            print(f"[!] Could not read {r_dir}: {e}")
    sftp.close()
    print("[+] All experimental artifacts downloaded.")


def main():
    ssh = get_client()
    try:
        setup_remote_env(ssh)
        sync_code(ssh)

        # Run 1: Distributional Policy Selection Benchmark
        run_cmd(
            ssh,
            f"cd {REMOTE_DIR} && source {VENV_DIR}/bin/activate && python pretraining/extra_trees_selector.py --benchmark-shift --use-distributional",
            "Experiment 3: Distributional Policy Selection Benchmark"
        )

        # Run 2: WUGNN vs Classical WMMSE Benchmark
        run_cmd(
            ssh,
            f"cd {REMOTE_DIR} && source {VENV_DIR}/bin/activate && python scripts/wugnn_benchmark.py --n-list 10 20 50 100 --n-repeats 30",
            "Experiment 4: WUGNN Speedup Benchmark"
        )

        # Run 3: MARL Ablation Stress Test (Realistic Channel + 3 seeds)
        run_cmd(
            ssh,
            f"cd {REMOTE_DIR} && source {VENV_DIR}/bin/activate && python agents/marl/marl_ablation_runner.py --steps 3000 --num-ues 8 --num-rbs 12 --num-episodes 3 --include-blockage --output-json data/marl_ablation_results.json",
            "Experiment 2: MARL Ablation Stress Test"
        )

        # Run 4: High-Step Convergence Run (100k timesteps)
        run_cmd(
            ssh,
            f"cd {REMOTE_DIR} && source {VENV_DIR}/bin/activate && python scripts/convergence_run.py --sla-profile balanced --max-timesteps 100000 --num-ues 8 --num-rbs 12 --eval-episodes 5 --save-json data/convergence_results.json --figure-path figures/fig4_convergence_curve.png",
            "Experiment 1: High-Step Convergence Run (100,000 steps)"
        )

        # Download all outputs
        download_results(ssh)

    finally:
        ssh.close()


if __name__ == "__main__":
    main()
