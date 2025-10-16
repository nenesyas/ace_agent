# local_test_runner.py
import subprocess, sys, os

def run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(">>>", " ".join(cmd))
    print(r.stdout)
    return r.returncode

workspace = "ace_workspace"
src_py = os.path.join(workspace, "main.py")
if not os.path.exists(src_py):
    print("NO_MAIN", src_py)
    sys.exit(1)

# tools list (we use module form when possible)
tools = ["ruff", "mypy", "bandit", "pytest"]
missing = []
for t in tools:
    # check availability via pip show
    if subprocess.run([sys.executable, "-m", "pip", "show", t], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        missing.append(t)
if missing:
    print("MISSING_TOOLS", ", ".join(missing))
    print("Install with: pip install " + " ".join(missing))

# run ruff as module for portability
if run([sys.executable, "-m", "ruff", "check", src_py]) != 0:
    sys.exit(2)
if run([sys.executable, "-m", "mypy", src_py]) != 0:
    sys.exit(3)
if run([sys.executable, "-m", "bandit", "-r", src_py]) != 0:
    sys.exit(4)
# run pytest if tests exist
if os.path.exists("tests"):
    if run([sys.executable, "-m", "pytest", "-q"]) != 0:
        sys.exit(5)
print("ALL TESTS PASS")
