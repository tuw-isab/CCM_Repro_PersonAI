import subprocess
import sys


def test():
    sys.exit(subprocess.run(["pytest"] + sys.argv[1:]).returncode)


def lint():
    sys.exit(subprocess.run(["ruff", "check", "."] + sys.argv[1:]).returncode)
