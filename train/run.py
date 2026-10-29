"""Run DRScaffold or a paper ablation, handing off adapters between stages."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from train.experiments import (BACKBONES, CONFIG_DIR, EXPERIMENTS, experiment_config,
                               load_config, stage_directory)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "train/configs/qwen2_5_vl_3b.json"


def cli_options(options):
    args = []
    for key, value in options.items():
        args.append(f"--{key}")
        values = value if isinstance(value, list) else [value]
        args.extend(str(x).lower() if isinstance(x, bool) else str(x) for x in values)
    return args


def last_checkpoint(directory):
    checkpoints = [p for p in directory.rglob("checkpoint-*")
                   if p.is_dir() and p.name.split("-")[-1].isdigit()
                   and (p / "adapter_config.json").is_file()
                   and (p / "adapter_model.safetensors").is_file()]
    if not checkpoints:
        raise ValueError(f"No completed LoRA checkpoint under {directory}")
    if len({p.parent for p in checkpoints}) != 1:
        raise ValueError(f"Multiple training runs found under {directory}")
    return max(checkpoints, key=lambda p: int(p.name.split("-")[-1]))


def phase_command(config, phase, data_dir, output_dir, adapter=None, resume=None):
    options = dict(config["common"])
    options.update(config["phases"][phase - 1])
    options.update(model=config["model"], seed=config["seed"], data_seed=config["seed"],
                   dataset=str(data_dir / ("p4_train.jsonl" if phase == 4 else "train.jsonl")),
                   val_dataset=str(data_dir / "val.jsonl"), output_dir=str(output_dir),
                   external_plugins=str(ROOT / "train/swift_plugin.py"),
                   loss_scale=f"drscaffold_p{phase}")
    if resume:
        options["resume_from_checkpoint"] = str(resume)
    elif adapter:
        options["adapters"] = str(adapter)
    return [sys.executable, "-m", "swift.cli.main", "sft", *cli_options(options)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=BACKBONES, default="qwen")
    parser.add_argument("--config", type=Path, help="Override the backbone configuration")
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="full")
    parser.add_argument("--model", help="Override the model ID with a local model directory")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/prepared")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start-step", "--start-phase", dest="start_phase", type=int, default=1)
    parser.add_argument("--end-step", "--end-phase", dest="end_phase", type=int)
    parser.add_argument("--adapter", type=Path, help="Previous step's adapter when starting after step 1")
    parser.add_argument("--resume", type=Path, help="Resume the starting stage, including optimizer state")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without data, models or GPUs")
    args = parser.parse_args()
    stages = EXPERIMENTS[args.experiment]
    args.end_phase = len(stages) if args.end_phase is None else args.end_phase
    if not 1 <= args.start_phase <= args.end_phase <= len(stages):
        parser.error("Require 1 <= start-step <= end-step <= number of scheduled stages")
    if args.adapter and args.resume:
        parser.error("Choose either --adapter or --resume")
    if args.start_phase == 1 and args.adapter:
        parser.error("--adapter is for a handoff after step 1")
    if args.start_phase > 1 and not (args.adapter or args.resume):
        parser.error("Starting after step 1 requires --adapter or --resume")
    config_path = args.config or CONFIG_DIR / BACKBONES[args.backbone]
    config = experiment_config(load_config(config_path), args.experiment)
    if args.model:
        config["model"] = args.model
    data_dir = args.data_dir.resolve()
    output_root = (args.output_dir or ROOT / "outputs" / config_path.stem / args.experiment).resolve()
    schedule = [(step, stages[step - 1]) for step in range(args.start_phase, args.end_phase + 1)]
    def destination_for(step, phase):
        return output_root / stage_directory(args.experiment, step, phase)
    adapter = args.adapter.resolve() if args.adapter else None
    resume = args.resume.resolve() if args.resume else None
    if resume and destination_for(*schedule[0]) not in resume.parents:
        parser.error("--resume must be a checkpoint inside the starting step output directory")
    if not args.dry_run:
        for name in ("train.jsonl", "val.jsonl") + (("p4_train.jsonl",) if any(phase == 4 for _, phase in schedule) else ()):
            if not (data_dir / name).is_file():
                parser.error(f"Missing {data_dir / name}; run python -m train.prepare first")
        for checkpoint in (adapter, resume):
            if checkpoint and not ((checkpoint / "adapter_config.json").is_file()
                                   and (checkpoint / "adapter_model.safetensors").is_file()):
                parser.error(f"Not a LoRA checkpoint: {checkpoint}")
        if resume and not (resume / "trainer_state.json").is_file():
            parser.error("Resume checkpoint has no trainer_state.json")
        # Check all destinations before starting an expensive training run.
        for step, phase in schedule:
            destination = destination_for(step, phase)
            if destination.exists() and any(destination.iterdir()) and not (resume and step == args.start_phase):
                parser.error(f"Output already exists: {destination}; use a new directory or explicit --resume")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    for step, phase in schedule:
        destination = destination_for(step, phase)
        command = phase_command(config, phase, data_dir, destination, adapter, resume)
        print(f"Step {step} / P{phase}: {shlex.join(command)}", flush=True)
        if args.dry_run:
            adapter = destination / "checkpoint-LAST"
        else:
            destination.mkdir(parents=True, exist_ok=True)
            manifest = {"experiment": args.experiment, "step": step, "phase": phase, "config": config, "command": command,
                        "distributed": {key: env.get(key) for key in
                                        ("CUDA_VISIBLE_DEVICES", "NPROC_PER_NODE", "MASTER_PORT")}}
            # Keep a separate record for every resume attempt.
            from datetime import datetime, timezone
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            (destination / f"launch-{stamp}.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            subprocess.run(command, cwd=ROOT, env=env, check=True)
            adapter = last_checkpoint(destination)
            (destination / "last_adapter.txt").write_text(str(adapter) + "\n", encoding="utf-8")
        resume = None
    if not args.dry_run:
        print(f"Final adapter: {adapter}")


if __name__ == "__main__":
    main()
