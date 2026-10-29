"""Launch independent Qwen stage ablations from the base model."""

import argparse
from pathlib import Path
import subprocess
import sys

from train.experiments import EXPERIMENTS
from train.run import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--model", help="Optional local Qwen2.5-VL-3B model directory")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/prepared")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/ablations")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(set(args.experiments)) != len(args.experiments):
        parser.error("Each experiment must appear only once")
    if not args.dry_run:
        for name in args.experiments:
            destination = args.output_dir / name
            if destination.exists() and any(destination.iterdir()):
                parser.error(f"Output already exists: {destination}; resume it with train.run")
    for name in args.experiments:
        command = [sys.executable, "-m", "train.run", "--experiment", name,
                   "--data-dir", str(args.data_dir.resolve()),
                   "--output-dir", str((args.output_dir / name).resolve())]
        if args.model:
            command += ["--model", args.model]
        if args.dry_run:
            command.append("--dry-run")
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
