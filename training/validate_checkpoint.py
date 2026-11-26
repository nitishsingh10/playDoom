"""Run the official validator against an export without replacing baseline weights."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights_dir", type=Path, help="Export directory containing checkpoint.pt and config.json")
    parser.add_argument("--map", action="append", default=None)
    parser.add_argument("--ticks", type=int, default=2100)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    weights = args.weights_dir.resolve()
    for name in ("checkpoint.pt", "config.json"):
        if not (weights / name).is_file():
            parser.error(f"Missing exported file: {weights / name}")

    # Short paths also avoid macOS AF_UNIX path-length limits in the validator.
    temp_root = "/tmp" if os.name == "posix" else None
    with tempfile.TemporaryDirectory(prefix="doom-export-", dir=temp_root) as temporary:
        stage = Path(temporary)
        shutil.copytree(
            root / "submission", stage / "submission",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "weights"),
        )
        target = stage / "submission" / "weights"
        target.mkdir()
        for name in ("checkpoint.pt", "config.json"):
            shutil.copy2(weights / name, target / name)
        command = [sys.executable, str(root / "validate.py"), "--repo", str(stage),
                   "--device", args.device, "--ticks", str(args.ticks)]
        for spec in args.map or [str(root / "maps" / "two_rooms.wad")]:
            command.extend(("--map", spec))
        environment = os.environ.copy()
        if temp_root:
            environment["TMPDIR"] = temp_root
        return subprocess.run(command, cwd=root, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
