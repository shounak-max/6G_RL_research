import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.0.24.7', port=2222, username='piyush', password='2£K40d#N', timeout=15)
ssh.get_transport().set_keepalive(15)

commands = [
    ("Check miniconda envs", "/home/piyush/miniconda3/bin/conda env list"),
    ("Check venv python", "/home/piyush/venv/bin/python -c \"import torch, gymnasium, scipy, sklearn, yaml, matplotlib, stable_baselines3; print('VENV_ALL_OK, PyTorch CUDA:', torch.cuda.is_available())\" 2>&1"),
]

for title, cmd in commands:
    print(f"=== {title} ===")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if out:
        print("OUT:", out)
    if err:
        print("ERR:", err)
    print()

ssh.close()
