import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.0.24.7', port=2222, username='piyush', password='2£K40d#N', timeout=15)
ssh.get_transport().set_keepalive(15)

commands = [
    ("Check conda / environments", "find /home/piyush -maxdepth 3 -name 'activate' 2>/dev/null || which conda || which mamba"),
    ("Check pip / python path", "which pip3 || which pip; find /home/piyush/.local -maxdepth 2 2>/dev/null"),
    ("Check internet connection", "curl -I -s --connect-timeout 5 https://pypi.org | head -n 3 || ping -c 2 8.8.8.8"),
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
