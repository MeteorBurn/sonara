"""One command for the delivery gate: every mandatory check, one verdict.

    .\\.venv\\Scripts\\python.exe scripts/run_gate.py [--base REV]

Runs, from the repo root and with the interpreter that launched it: the core
tests in both build configurations, the bindings build, the Python suites and
the fidelity gates. Prints one line per step and a verdict; on the first
failure it prints the tail of that step's output and exits 1.

`--base` is the revision the fidelity gates compare against. The default
`HEAD` fits uncommitted work; after committing, pass the commit before it.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(sys.executable).parent


def steps(base):
    python = sys.executable
    return [
        ("core tests", ["cargo", "test", "-p", "sonara"]),
        ("core tests, optional descriptors", ["cargo", "test", "-p", "sonara", "--features", "aggression"]),
        ("bindings build", [str(SCRIPTS / "maturin"), "develop", "--release", "--locked"]),
        ("python suites", [python, "scripts/run_python_tests.py"]),
        ("fidelity gates", [python, "scripts/run_fidelity_gate.py", "--base", base]),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="HEAD")
    base = parser.parse_args().base
    started = time.monotonic()
    for name, command in steps(base):
        t = time.monotonic()
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(f"{'ok  ' if run.returncode == 0 else 'FAIL'}  {name}  ({time.monotonic() - t:.0f} s)", flush=True)
        if run.returncode != 0:
            print("\n".join((run.stdout + run.stderr).splitlines()[-40:]))
            print(f"GATE FAILED at: {name}")
            return 1
        if name == "bindings build":
            # maturin leaves the debug symbols beside the module; they are not part of the package
            (ROOT / "python" / "sonara" / "sonara.pdb").unlink(missing_ok=True)
    print(f"GATE PASSED  ({time.monotonic() - started:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
