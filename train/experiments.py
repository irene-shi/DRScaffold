"""Paper stage schedules and shared backbone configuration."""

from copy import deepcopy
import json
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
BACKBONES = {
    "qwen": "qwen2_5_vl_3b.json",
    "internvl": "internvl2_5_2b.json",
    "phi4": "phi4_multimodal.json",
}
EXPERIMENTS = {
    "full": (1, 2, 3, 4),
    "s4": (4,),
    "s34": (3, 4),
    "s234": (2, 3, 4),
    "s134": (1, 3, 4),
    "reverse": (4, 3, 2, 1),
    "random_1324": (1, 3, 2, 4),
    "random_3412": (3, 4, 1, 2),
    "random_4123": (4, 1, 2, 3),
}


def load_config(path, seen=()):
    path = Path(path).resolve()
    if path in seen:
        raise ValueError(f"Circular config inheritance: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    parent = config.pop("extends", None)
    if parent:
        base = load_config(path.parent / parent, (*seen, path))
        base["common"].update(config.pop("common", {}))
        base.update(config)
        config = base
    if len(config["phases"]) != 4:
        raise ValueError("Config must define exactly four phases")
    return config


def experiment_config(config, experiment):
    """Redistribute removed epochs proportionally; retain each stage's LR/mask."""
    result = deepcopy(config)
    stages = EXPERIMENTS[experiment]
    epochs = [float(phase["num_train_epochs"]) for phase in result["phases"]]
    if any(value <= 0 for value in epochs):
        raise ValueError("Stage epochs must be positive")
    factor = sum(epochs) / sum(epochs[phase - 1] for phase in stages)
    for phase in stages:
        result["phases"][phase - 1]["num_train_epochs"] *= factor
    return result


def stage_directory(experiment, step, phase):
    return f"p{phase}" if experiment == "full" else f"step{step}_p{phase}"
