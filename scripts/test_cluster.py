import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.0.24.7', port=2222, username='piyush', password='2£K40d#N', timeout=15)
ssh.get_transport().set_keepalive(15)

commands = [
    ("Python Version", "python3 --version"),
    ("CUDA Check", "python3 -c \"import torch; print('Torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available(), 'Devices:', torch.cuda.device_count())\""),
    ("Gymnasium Check", "python3 -c \"import gymnasium; print('Gymnasium:', gymnasium.__version__)\""),
    ("SB3 Check", "python3 -c \"import stable_baselines3; print('SB3:', stable_baselines3.__version__)\""),
    ("SciPy & Scikit-Learn", "python3 -c \"import scipy, sklearn, matplotlib, yaml; print('SciPy, sklearn, matplotlib, pyyaml all OK')\""),
    ("Disk Space & RAM", "df -h . && free -h"),
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
