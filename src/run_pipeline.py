"""
Deprecated wrapper — forwards to pipeline_real.py.

Prefer: python src/pipeline_real.py [--full | --quick]
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "src" / "pipeline_real.py"), *sys.argv[1:]], cwd=root, check=True)


if __name__ == "__main__":
    main()
